import logging
import os
from typing import Dict, List, Optional, Tuple

import dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (Application, CallbackContext, CallbackQueryHandler,
                          CommandHandler, MessageHandler, filters)
from langchain_community.chat_models import GigaChat

# =============================
# Configuration and Globals
# =============================

dotenv.load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_BASE: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
print(TELEGRAM_BOT_TOKEN)
# Load all OPENROUTER_API_KEY-like env vars: OPENROUTER_API_KEY, OPENROUTER_API_KEY2, OPENROUTER_API_KEY3, ...
# Also support a comma-separated OPENROUTER_API_KEYS if provided
_api_keys_from_single_var = [k.strip() for k in os.getenv("OPENROUTER_API_KEYS", "").split(",") if k.strip()]
_api_keys_from_numbered_vars: List[str] = []
for env_key, env_val in os.environ.items():
    if env_key.startswith("OPENROUTER_API_KEY") and env_val and env_val.strip():
        _api_keys_from_numbered_vars.append(env_val.strip())

OPENROUTER_API_KEYS: List[str] = [*dict.fromkeys([*_api_keys_from_single_var, *_api_keys_from_numbered_vars])]

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN отсутствует в .env")
if not OPENROUTER_API_KEYS:
    raise RuntimeError("Не найдены ключи OpenRouter. Добавьте OPENROUTER_API_KEY (и/или нумерованные OPENROUTER_API_KEY2,3,...) или OPENROUTER_API_KEYS")

# Models (modes)
MODELS: Dict[str, str] = {
    "ChatGPT-4o": "openai/gpt-4o",
    "ChatGPT-4o-mini": "openai/gpt-4o-mini", 
    "Gemini-2.5-Flash": "google/gemini-2.5-flash",
    # "Claude-3.5-Sonnet": "anthropic/claude-3.5-sonnet",
    "GROK-4-Fast": "x-ai/grok-4-fast",
    "GigaChat": "gigachat/gigachat",
}
DEFAULT_MODEL_KEY: str = "ChatGPT-4o"
# оставляем только ключ для платных моделей
OPENROUTER_API_KEY_NO_FREE: str = os.getenv("OPENROUTER_API_KEY_NO_FREE", "")
OPENROUTER_API_KEYS = [OPENROUTER_API_KEY_NO_FREE]
GIGA_CHAT_KEY = os.getenv("GIGA_CHAT_KEY", "")
# In-memory per-chat state
chat_id_to_model_key: Dict[int, str] = {}
chat_id_to_key_index: Dict[int, int] = {}
chat_id_to_history: Dict[int, List[Dict[str, str]]] = {}  # Хранение истории диалогов

with open("архитектура.txt", "r", encoding="utf-8") as f:
    base_knowledge = f.read()


PROMPT_TEMPLATE: str = """
Ты опытный специалист по памятникам архитектуры и культурному наследию России.
Вот информация о памятниках архитектуры и культурном наследии России: {base_knowledge}.
Твоя задача: проконсультировать пользователя и ответить на вопросы о памятниках архитектуры и культурном наследии России.
Следи за чистотой русского языка и грамотностью! Это ОЧЕНЬ важно!
Не отвечай на вопросы, которые не относятся к памятникам архитектуры и культурному наследии России.
Если пользователь задает вопрос, который не относится к памятникам архитектуры и культурному наследии России, скажи, что ты специалист по этой теме и можешь ответить только на вопросы, которые относятся к памятникам архитектуры и культурному наследии России.
Будь добрым, вежливым и приветливым!
"""

# =============================
# Utility functions
# =============================


# def get_display_examples() -> str:
#     if training_df is None or training_df.empty:
#         return ""
#     # Keeping the same spirit as the notebook: include the DF string
#     return training_df.to_string(index=False)


def get_current_model_key(chat_id: int) -> str:
    return chat_id_to_model_key.get(chat_id, DEFAULT_MODEL_KEY)


def set_current_model_key(chat_id: int, model_key: str) -> None:
    chat_id_to_model_key[chat_id] = model_key


def get_current_key_index(chat_id: int) -> int:
    return chat_id_to_key_index.get(chat_id, 0) % len(OPENROUTER_API_KEYS)


def set_current_key_index(chat_id: int, index: int) -> None:
    chat_id_to_key_index[chat_id] = index % len(OPENROUTER_API_KEYS)


def get_chat_history(chat_id: int) -> List[Dict[str, str]]:
    """Получить историю диалога для чата."""
    return chat_id_to_history.get(chat_id, [])


def add_to_history(chat_id: int, role: str, content: str) -> None:
    """Добавить сообщение в историю диалога."""
    if chat_id not in chat_id_to_history:
        chat_id_to_history[chat_id] = []
    chat_id_to_history[chat_id].append({"role": role, "content": content})


def clear_chat_history(chat_id: int) -> None:
    """Очистить историю диалога для чата."""
    if chat_id in chat_id_to_history:
        chat_id_to_history[chat_id] = []


def build_system_prompt() -> str:
    return PROMPT_TEMPLATE.format(base_knowledge=base_knowledge)


def get_fallback_models(current_model: str) -> List[str]:
    """Возвращает список моделей для fallback в порядке приоритета."""
    # Определяем порядок приоритета моделей
    model_priority = [
        "GigaChat",           # Российская модель - приоритет
        "ChatGPT-4o",         # OpenAI - надежная
        "ChatGPT-4o-mini",    # OpenAI - быстрая
        # "Claude-3.5-Sonnet", # Anthropic - качественная
        "Gemini-2.5-Flash",  # Google - быстрая
        "GROK-4-Fast"        # X.AI - альтернативная
    ]
    
    # Исключаем текущую модель и возвращаем остальные
    fallback_models = [model for model in model_priority if model != current_model]
    return fallback_models


def is_rate_or_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    keywords = [
        "rate limit", "429", "quota", "payment required", "402",
        "too many requests", "insufficient_quota", "credit", "over limit",
    ]
    return any(k in message for k in keywords)


async def call_llm_with_rotation(chat_id: int, model_key: str, user_text: str) -> Tuple[str, Optional[str]]:
    """Call LLM; rotate keys on quota/rate errors. Returns (content, rotated_info)."""
    rotated_info: Optional[str] = None
    total_keys = len(OPENROUTER_API_KEYS)
    start_index = get_current_key_index(chat_id)

    for attempt in range(total_keys):
        key_index = (start_index + attempt) % total_keys
        set_current_key_index(chat_id, key_index)
        api_key = OPENROUTER_API_KEYS[key_index]

        try:
            model_id = MODELS[model_key]
            if model_key == "GigaChat":
                llm = GigaChat(credentials=GIGA_CHAT_KEY,
                verify_ssl_certs=False,
                scope="GIGACHAT_API_PERS",
                temperature=0.5,
                max_tokens=1000)
            else:
                llm = ChatOpenAI(
                    openai_api_key=api_key,
                    openai_api_base=OPENROUTER_API_BASE,
                    model_name=model_id,
                    temperature=0.5,
                    max_tokens=1000,
                )
            system_prompt = build_system_prompt()
            # Используем правильный формат для LangChain с учетом истории
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            messages = [SystemMessage(content=system_prompt)]
            
            # Добавляем историю диалога
            history = get_chat_history(chat_id)
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            # Добавляем текущее сообщение пользователя
            messages.append(HumanMessage(content=user_text))
            
            result = await llm.ainvoke(messages)
            return str(result.content), rotated_info
        except Exception as exc:  # noqa: BLE001
            if is_rate_or_quota_error(exc):
                next_index = (key_index + 1) % total_keys
                if next_index != key_index:
                    set_current_key_index(chat_id, next_index)
                    rotated_info = "Исчерпан лимит текущего ключа, переключаюсь на следующий ключ..."
                    continue
            # Non-quota error or no more keys
            raise

    raise RuntimeError("Не удалось выполнить запрос: закончились доступные ключи.")


async def call_llm_with_fallback(chat_id: int, model_key: str, user_text: str) -> Tuple[str, Optional[str]]:
    """Call LLM with fallback to other models if empty response. Returns (content, fallback_info)."""
    fallback_info: Optional[str] = None
    
    # Сначала пробуем основную модель
    try:
        content, rotated_info = await call_llm_with_rotation(chat_id, model_key, user_text)
        
        # Проверяем, что ответ не пустой
        if content and content.strip():
            return content, rotated_info
        
        # Если ответ пустой, пробуем fallback модели
        fallback_models = get_fallback_models(model_key)
        logging.info(f"Получен пустой ответ от модели {model_key}, пробуем fallback модели: {fallback_models}")
        
        for fallback_model in fallback_models:
            try:
                logging.info(f"Пробуем fallback модель: {fallback_model}")
                content, _ = await call_llm_with_rotation(chat_id, fallback_model, user_text)
                
                if content and content.strip():
                    fallback_info = f"🔄 Модель {model_key} вернула пустой ответ, переключился на {fallback_model}"
                    set_current_model_key(chat_id, fallback_model)  # Обновляем текущую модель
                    return content, fallback_info
                else:
                    logging.warning(f"Fallback модель {fallback_model} также вернула пустой ответ")
                    continue
                    
            except Exception as exc:
                logging.warning(f"Ошибка при использовании fallback модели {fallback_model}: {exc}")
                continue
        
        # Если все модели вернули пустые ответы
        return "", "❌ Все доступные модели вернули пустые ответы. Попробуйте переформулировать вопрос."
        
    except Exception as exc:
        # Если основная модель упала с ошибкой, пробуем fallback
        fallback_models = get_fallback_models(model_key)
        logging.info(f"Ошибка в основной модели {model_key}: {exc}, пробуем fallback модели: {fallback_models}")
        
        for fallback_model in fallback_models:
            try:
                logging.info(f"Пробуем fallback модель после ошибки: {fallback_model}")
                content, _ = await call_llm_with_rotation(chat_id, fallback_model, user_text)
                
                if content and content.strip():
                    fallback_info = f"🔄 Модель {model_key} недоступна, переключился на {fallback_model}"
                    set_current_model_key(chat_id, fallback_model)  # Обновляем текущую модель
                    return content, fallback_info
                else:
                    logging.warning(f"Fallback модель {fallback_model} вернула пустой ответ")
                    continue
                    
            except Exception as exc_fallback:
                logging.warning(f"Ошибка при использовании fallback модели {fallback_model}: {exc_fallback}")
                continue
        
        # Если все fallback модели тоже упали
        raise RuntimeError(f"Не удалось получить ответ ни от одной модели. Последняя ошибка: {exc}")


def build_modes_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="ChatGPT-4o", callback_data="set_mode:ChatGPT-4o")],
        [InlineKeyboardButton(text="ChatGPT-4o-mini", callback_data="set_mode:ChatGPT-4o-mini")],
        [InlineKeyboardButton(text="Gemini-2.5-Flash", callback_data="set_mode:Gemini-2.5-Flash")],
        # [InlineKeyboardButton(text="Claude-3.5-Sonnet", callback_data="set_mode:Claude-3.5-Sonnet")],
        [InlineKeyboardButton(text="GROK-4-Fast", callback_data="set_mode:GROK-4-Fast")],
        [InlineKeyboardButton(text="GigaChat", callback_data="set_mode:GigaChat")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_control_panel() -> ReplyKeyboardMarkup:
    """Создает постоянную панель управления ботом."""
    keyboard = [
        [KeyboardButton("📖 Инструкция"), KeyboardButton("🤖 Сменить модель")],
        [KeyboardButton("ℹ️ Текущая модель"), KeyboardButton("🆕 Новый диалог")],
        [KeyboardButton("❓ Помощь")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


# =============================
# Handlers
# =============================


async def start(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_chat.id
    set_current_model_key(chat_id, get_current_model_key(chat_id))
    
    instruction_text = """
🏛️ <b>Добро пожаловать в бот-консультант по памятникам архитектуры и культурному наследию России!</b>

📋 <b>Инструкция по использованию:</b>

1️⃣ <b>Выбор модели AI</b>
   Нажмите кнопку "🤖 Сменить модель" или используйте команду /mode
   Доступные модели: ChatGPT-4o, ChatGPT-4o-mini, Gemini-2.5-Flash, GROK-4-Fast, GigaChat

2️⃣ <b>Задайте вопрос</b>
   Просто напишите свой вопрос о памятниках архитектуры и культурном наследии России
   Например: "Расскажи о Кремле" или "Какие памятники есть в Санкт-Петербурге?"

3️⃣ <b>Получите ответ</b>
   Бот обработает ваш запрос и предоставит подробную консультацию
   Бот запоминает историю вашего диалога и учитывает контекст предыдущих сообщений

4️⃣ <b>Новый диалог</b>
   Нажмите кнопку "🆕 Новый диалог" или используйте команду /new
   Это очистит историю диалога и начнет новую беседу с чистого листа

💡 <b>Панель управления:</b>
   📖 Инструкция - показать эту инструкцию снова
   🤖 Сменить модель - выбрать другую AI модель
   ℹ️ Текущая модель - узнать, какая модель сейчас активна
   🆕 Новый диалог - начать новую беседу с очисткой истории
   ❓ Помощь - получить дополнительную помощь

⚠️ <b>Важно:</b> Бот специализируется только на вопросах о памятниках архитектуры и культурном наследии России.

Готовы начать? Задайте свой первый вопрос! 👇
"""
    
    await update.message.reply_text(
        instruction_text,
        parse_mode=ParseMode.HTML,
        reply_markup=build_control_panel(),
    )


async def mode(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_chat.id
    current_key = get_current_model_key(chat_id)
    await update.message.reply_text(
        f"Текущий режим: {current_key}. Выберите другой:",
        reply_markup=build_modes_keyboard(),
    )


async def on_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data.startswith("set_mode:"):
        _, model_key = data.split(":", 1)
        if model_key in MODELS:
            set_current_model_key(query.message.chat.id, model_key)
            await query.edit_message_text(
                text=f"Режим установлен: {model_key} \n\nЗадайте свой вопрос!",
                parse_mode=ParseMode.HTML,
            )
        else:
            await query.edit_message_text("Неизвестный режим")


async def new_dialog(update: Update, context: CallbackContext) -> None:
    """Начать новый диалог с очисткой истории."""
    chat_id = update.effective_chat.id
    history_length = len(get_chat_history(chat_id))
    clear_chat_history(chat_id)
    
    await update.message.reply_text(
        f"🆕 <b>Начат новый диалог!</b>\n\n"
        f"История предыдущего диалога очищена (было сообщений: {history_length}).\n"
        f"Можете задавать новые вопросы с чистого листа! 💬",
        parse_mode=ParseMode.HTML
    )


async def help_command(update: Update, context: CallbackContext) -> None:
    """Обработчик команды помощи."""
    help_text = """
❓ <b>Помощь по боту</b>

<b>Доступные команды:</b>
/start - Начать работу с ботом и показать инструкцию
/mode - Изменить модель AI
/new - Начать новый диалог (очистить историю)
/help - Показать это сообщение помощи

<b>Как использовать бот:</b>
• Выберите AI модель через панель управления или команду /mode
• Задайте вопрос о памятниках архитектуры России
• Получите подробный ответ от выбранной AI модели
• Бот запоминает историю диалога и учитывает контекст

<b>Примеры вопросов:</b>
• "Расскажи о Московском Кремле"
• "Какие архитектурные памятники есть в Казани?"
• "История Петергофа"
• "Что такое объект культурного наследия?"

<b>Технические вопросы:</b>
Если у вас возникли проблемы с ботом, попробуйте:
1. Начать новый диалог командой /new
2. Перезапустить бота командой /start
3. Сменить модель через /mode
4. Проверить интернет-соединение

Удачи в изучении культурного наследия России! 🏛️
"""
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


async def show_current_model(update: Update, context: CallbackContext) -> None:
    """Показывает текущую активную модель."""
    chat_id = update.effective_chat.id
    current_key = get_current_model_key(chat_id)
    await update.message.reply_text(
        f"ℹ️ <b>Текущая модель:</b> {current_key}\n\n"
        f"Эта модель будет использоваться для обработки ваших запросов.\n"
        f"Чтобы сменить модель, нажмите кнопку '🤖 Сменить модель' или используйте /mode",
        parse_mode=ParseMode.HTML
    )


async def on_text(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    
    # Обработка кнопок панели управления
    if text == "📖 Инструкция":
        await start(update, context)
        return
    elif text == "🤖 Сменить модель":
        await mode(update, context)
        return
    elif text == "ℹ️ Текущая модель":
        await show_current_model(update, context)
        return
    elif text == "🆕 Новый диалог":
        await new_dialog(update, context)
        return
    elif text == "❓ Помощь":
        await help_command(update, context)
        return
    
    # Обработка обычного текстового запроса
    model_key = get_current_model_key(chat_id)
    
    # Отправляем сообщение о принятии запроса в работу
    processing_message = await update.message.reply_text(
        "⏳ <b>Запрос принят в работу...</b>\n"
        f"Используется модель: {model_key}\n"
        "Пожалуйста, подождите, идет обработка запроса.",
        parse_mode=ParseMode.HTML
    )

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        content, fallback_info = await call_llm_with_fallback(chat_id, model_key, text)
        
        # Удаляем сообщение о обработке
        try:
            await processing_message.delete()
        except Exception:
            # Игнорируем ошибки при удалении сообщения
            pass
        
        # Проверяем, что контент не пустой
        if not content or not content.strip():
            await update.message.reply_text(
                "❌ <b>Не удалось получить ответ от AI моделей</b>\n\n"
                "Попробуйте переформулировать вопрос или смените модель.",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Добавляем сообщение пользователя и ответ бота в историю
        add_to_history(chat_id, "user", text)
        add_to_history(chat_id, "assistant", content)
        
        # Показываем информацию о переключении модели, если было
        if fallback_info:
            await update.message.reply_text(fallback_info)
        
        await update.message.reply_text(content)
    except Exception as exc:  # noqa: BLE001
        logging.exception("LLM error")
        # Удаляем сообщение о обработке в случае ошибки
        try:
            await processing_message.delete()
        except Exception:
            # Игнорируем ошибки при удалении сообщения
            pass
        await update.message.reply_text(
            f"❌ <b>Ошибка генерации:</b>\n{exc}\n\n"
            "Попробуйте еще раз или смените модель.",
            parse_mode=ParseMode.HTML
        )


# =============================
# Entrypoint
# =============================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("mode", mode))
    application.add_handler(CommandHandler("new", new_dialog))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

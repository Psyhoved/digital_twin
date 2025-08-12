import logging
import os
from typing import Dict, List, Optional, Tuple

import dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (Application, CallbackContext, CallbackQueryHandler,
                          CommandHandler, MessageHandler, filters)


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
    "qwen_coder": "qwen/qwen3-coder:free",
    "deepseek_r1": "deepseek/deepseek-r1-0528:free",
    "deepseek_chat": "deepseek/deepseek-chat-v3-0324:free",
    "chimera_r1t2": "tngtech/deepseek-r1t2-chimera:free",
    "ds_distill_l70b": "deepseek/deepseek-r1-distill-llama-70b:free",
}
DEFAULT_MODEL_KEY: str = "deepseek_r1"

# In-memory per-chat state
chat_id_to_model_key: Dict[int, str] = {}
chat_id_to_key_index: Dict[int, int] = {}

# Load training examples (same as notebook logic uses df)
EXCEL_PATH: str = os.getenv("EXAMPLES_XLSX_PATH", "База_знаний.xlsx")
try:
    training_df: pd.DataFrame = pd.read_excel(EXCEL_PATH)
except Exception:
    training_df = pd.DataFrame()

PROMPT_TEMPLATE: str = (
    """
Ты опытный копирайтер с 12 летним стажем. Твоя задача на основе текста задачи (по сути это текст, описывающий тему) написать цепляющий пост в соцсети.
Проанализируй структуру текста, предложения, потенциальную аудиторию и так далее. Используй это, чтобы сделать результат более увлекательным.
Вот список примеров, как нужно писать посты на основе задачи: {examples}.
Задача будет приходить под флагом user promt.
Сделай свой текст МАКСИМАЛЬНО насыщенным и интересным для читателей!
Всегда отвечай на красивом чистом русском языке!
Следи за чистотой русского языка и грамотностью! Это ОЧЕНЬ важно!
Не давай от себя никаких комментариев, ты должен дать только текст самого поста и ничего больше.
"""
).strip()


# =============================
# Utility functions
# =============================


def get_display_examples() -> str:
    if training_df is None or training_df.empty:
        return ""
    # Keeping the same spirit as the notebook: include the DF string
    return training_df.to_string(index=False)


def get_current_model_key(chat_id: int) -> str:
    return chat_id_to_model_key.get(chat_id, DEFAULT_MODEL_KEY)


def set_current_model_key(chat_id: int, model_key: str) -> None:
    chat_id_to_model_key[chat_id] = model_key


def get_current_key_index(chat_id: int) -> int:
    return chat_id_to_key_index.get(chat_id, 0) % len(OPENROUTER_API_KEYS)


def set_current_key_index(chat_id: int, index: int) -> None:
    chat_id_to_key_index[chat_id] = index % len(OPENROUTER_API_KEYS)


def build_system_prompt() -> str:
    return PROMPT_TEMPLATE.format(examples=get_display_examples())


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
            llm = ChatOpenAI(
                openai_api_key=api_key,
                openai_api_base=OPENROUTER_API_BASE,
                model_name=model_id,
            )
            system_prompt = build_system_prompt()
            composed_input = (
                f"system promt: {system_prompt}, user promt: {user_text}"
            )
            result = await llm.ainvoke(composed_input)
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


def build_modes_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="Qwen Coder", callback_data="set_mode:qwen_coder")],
        [InlineKeyboardButton(text="DeepSeek R1", callback_data="set_mode:deepseek_r1")],
        [InlineKeyboardButton(text="DeepSeek Chat", callback_data="set_mode:deepseek_chat")],
        [InlineKeyboardButton(text="Chimera R1T2", callback_data="set_mode:chimera_r1t2")],
        [InlineKeyboardButton(text="DS Distill L70B", callback_data="set_mode:ds_distill_l70b")],
    ]
    return InlineKeyboardMarkup(buttons)


# =============================
# Handlers
# =============================


async def start(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_chat.id
    set_current_model_key(chat_id, get_current_model_key(chat_id))
    await update.message.reply_text(
        "Привет! Я бот для генерации постов. Выберите модель в меню ниже или командой /mode.",
        reply_markup=build_modes_keyboard(),
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
                text=f"Режим установлен: {model_key}",
                parse_mode=ParseMode.HTML,
            )
        else:
            await query.edit_message_text("Неизвестный режим")


async def on_text(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_chat.id
    model_key = get_current_model_key(chat_id)
    text = update.message.text.strip()

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        content, rotated_info = await call_llm_with_rotation(chat_id, model_key, text)
        if rotated_info:
            await update.message.reply_text(rotated_info)
        await update.message.reply_text(content)
    except Exception as exc:  # noqa: BLE001
        logging.exception("LLM error")
        await update.message.reply_text(f"Ошибка генерации: {exc}")


# =============================
# Entrypoint
# =============================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("mode", mode))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

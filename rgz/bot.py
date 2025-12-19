import asyncio
import logging
import json
import io
from pathlib import Path
from typing import Optional
from datetime import datetime

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    FSInputFile,
    BufferedInputFile,
)
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

from config import TELEGRAM_TOKEN, BASE_DIR
from rag_service import RagService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

load_dotenv()

user_state: dict[int, dict] = {}


def get_user_state(user_id: int) -> dict:
    if user_id not in user_state:
        user_state[user_id] = {
            "history": [],
            "favorites": [],
            "last_answer": None,
            "last_sources": [],
            "awaiting": None,
        }
    return user_state[user_id]


def set_user_state(user_id: int, **kwargs):
    state = get_user_state(user_id)
    for k, v in kwargs.items():
        if v is None and k in state:
            if k in ("history", "favorites", "last_sources"):
                continue
            state[k] = None
        elif v is not None:
            state[k] = v


def add_to_history(user_id: int, question: str, answer: str):
    state = get_user_state(user_id)
    state["history"].append((question, answer))
    if len(state["history"]) > 10:
        state["history"] = state["history"][-10:]


def add_to_favorites(user_id: int, answer: str):
    state = get_user_state(user_id)
    state["favorites"].append({"text": answer, "date": datetime.now().isoformat()})
    if len(state["favorites"]) > 20:
        state["favorites"] = state["favorites"][-20:]


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="💾 Избранное"), KeyboardButton(text="📤 Экспорт")],
            [KeyboardButton(text="🔄 Сброс")],
        ],
        resize_keyboard=True,
    )

def feedback_inline_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍", callback_data="fb:good"),
                InlineKeyboardButton(text="👎", callback_data="fb:bad"),
                InlineKeyboardButton(text="💾 Сохранить", callback_data="fb:save"),
            ]
        ]
    )




def export_history_txt(state: dict) -> bytes:
    lines = ["=== Образовательный ассистент: история запросов ===\n"]
    for i, (q, a) in enumerate(state.get("history", []), 1):
        lines.append(f"--- Запрос {i} ---")
        lines.append(f"Вопрос: {q}")
        lines.append(f"Ответ:\n{a}\n")
    return "\n".join(lines).encode("utf-8")


def export_favorites_txt(state: dict) -> bytes:
    lines = ["=== Избранное ===\n"]
    for i, fav in enumerate(state.get("favorites", []), 1):
        lines.append(f"--- {i}. {fav.get('date', '')} ---")
        lines.append(fav.get("text", ""))
        lines.append("")
    return "\n".join(lines).encode("utf-8")




async def handle_question(message: Message, rag: RagService, question: str = None, user_id: int = None):
    question = question or (message.text or "").strip()
    if not question:
        await message.reply("Пришлите текстовый вопрос.")
        return

    uid = user_id or message.from_user.id
    state = get_user_state(uid)

    await message.answer("Думаю над ответом... ⏳", parse_mode=None)

    try:
        answer, docs = await asyncio.to_thread(
            rag.generate_answer,
            question,
        )
        
        set_user_state(uid, last_answer=answer, last_sources=[d[1].get("source", "") for d in docs])
        add_to_history(uid, question, answer)
        
        sources = list(set(d[1].get("source", "") for d in docs if d[1].get("source")))
        sources_text = ""
        if sources:
            sources_text = "\n\n📚 Источники: " + ", ".join(sources[:3])
        
        full_answer = answer + sources_text
        
        try:
            await message.answer(full_answer, parse_mode=ParseMode.MARKDOWN, reply_markup=feedback_inline_keyboard())
        except Exception:
            await message.answer(full_answer, parse_mode=None, reply_markup=feedback_inline_keyboard())
        
        
    except Exception as exc:
        logging.exception("Failed to answer question")
        try:
            await message.answer(f"Ошибка: {exc}", parse_mode=None)
        except Exception:
            pass


async def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set")

    bot = Bot(
        token=TELEGRAM_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )
    dp = Dispatcher()
    rag = RagService()

    @dp.message(CommandStart())
    async def start_handler(message: types.Message):
        welcome = (
            "👋 *Привет! Я образовательный ассистент.*\n\n"
            "Я помогу тебе с обучением, отвечая на вопросы по материалам.\n\n"
            "*Как пользоваться:*\n"
            "Просто задай вопрос!\n\n"
            "*Команды:*\n"
            "/help — подробная справка\n"
            "/fav — избранное\n\n"
            "Начнём?"
        )
        await message.answer(welcome, reply_markup=main_keyboard())

    @dp.message(Command("help"))
    async def help_handler(message: types.Message):
        help_text = (
            "📖 *Справка по боту*\n\n"
            "*Возможности:*\n"
            "• Отвечаю на вопросы по загруженным материалам.\n"
            "• Могу суммировать тексты, объяснять понятия.\n\n"
            "*Команды:*\n"
            "/start — начать работу\n"
            "/fav — избранное\n\n"
            "Просто напиши мне свой вопрос!"
        )
        await message.answer(help_text, reply_markup=main_keyboard())


    @dp.message(Command("fav"))
    @dp.message(F.text == "💾 Избранное")
    async def fav_handler(message: types.Message):
        state = get_user_state(message.from_user.id)
        favs = state.get("favorites", [])
        if not favs:
            await message.answer("💾 Избранное пусто. Нажмите 💾 под ответом, чтобы сохранить.", parse_mode=None)
            return
        text = "💾 Ваше избранное:\n\n"
        for i, fav in enumerate(favs[-5:], 1):
            snippet = fav.get("text", "")[:200] + "..."
            text += f"{i}. {snippet}\n\n"
        try:
            await message.answer(text, parse_mode=None)
        except Exception:
            await message.answer("Не удалось показать избранное.", parse_mode=None)

    @dp.message(F.text == "📤 Экспорт")
    async def export_handler(message: types.Message):
        state = get_user_state(message.from_user.id)
        history = state.get("history", [])
        if not history:
            await message.answer("История пуста. Задайте несколько вопросов, потом сможете экспортировать.")
            return
        content = export_history_txt(state)
        doc = BufferedInputFile(content, filename="edu_assistant_history.txt")
        await message.answer_document(doc, caption="📤 Ваша история запросов")

    @dp.message(F.text == "🔄 Сброс")
    async def reset_handler(message: types.Message):
        set_user_state(
            message.from_user.id,
            awaiting=None,
        )
        await message.answer("🔄 Параметры сброшены.", reply_markup=main_keyboard())

    @dp.callback_query(F.data.startswith("fb:"))
    async def feedback_callback(callback: CallbackQuery):
        action = callback.data.split(":", 1)[1]
        state = get_user_state(callback.from_user.id)
        if action == "good":
            await callback.answer("Спасибо за отзыв! 👍")
        elif action == "bad":
            await callback.answer("Жаль, что не помогло. Попробуйте уточнить запрос.")
        elif action == "save":
            if state.get("last_answer"):
                add_to_favorites(callback.from_user.id, state["last_answer"])
                await callback.answer("💾 Сохранено в избранное!")
            else:
                await callback.answer("Нечего сохранять.")


    @dp.message()
    async def generic_handler(message: types.Message):
        await handle_question(message, rag)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

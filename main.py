import os
from io import BytesIO
from time import ctime

import numpy as np
import redis
from aiogram import Bot, Dispatcher, types
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from PIL import Image

from app.redis_client import (create_index, load_data_to_redis, load_mnist,
                              predict_label)

with open(".tg_bot_token", "r") as f:
    tg_token = f.readline()

bot = Bot(token=tg_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)

images, labels = load_mnist()


@dp.message(Command("init"))
async def initialize_index(message: types.Message):
    """
    Initialize the Redis index with default MNIST data.
    """
    try:
        redis_client.flushdb()  # clean the redis db

        load_data_to_redis(redis_client, images.tolist(), labels.tolist())
        await message.answer("Creating HNSW Index...")

        create_index(redis_client)
        await message.answer("Redis index initialized successfully with MNIST data!")
    except Exception as e:
        print(ctime(), f"Error initializing Redis index: {e}")
        await message.answer("Failed to initialize Redis index.")


@dp.message(Command("add"))
async def add_custom_image(message: types.Message):
    """
    Add a custom image with a label to the Redis index.
    """
    if not message.reply_to_message or not (
        message.reply_to_message.photo or message.reply_to_message.document
    ):
        await message.answer(
            "Please reply to an image (photo or document) and include the label in the command (e.g., /add <label>)."
        )
        return

    try:
        label = (
            message.text.split(maxsplit=1)[1]
            if len(message.text.split(maxsplit=1)) > 1
            else None
        )
        if not label:
            await message.answer("Please provide a label for the image.")
            return

        if label not in [str(x) for x in range(10)]:
            await message.answer("Label must be 0-9")
            return

        file_id = (
            message.reply_to_message.photo[-1].file_id
            if message.reply_to_message.photo
            else message.reply_to_message.document.file_id
        )
        file = await bot.download(file_id)

        image = np.array(Image.open(BytesIO(file.read())).convert("L"))
        if image.shape != (8, 8):
            await message.answer("Image must be 8x8")
            return

        image = image.flatten()

        load_data_to_redis(redis_client, [image.astype(np.float16).tolist()], [label])
        await message.answer(f"Image with label '{label}' added to the Redis index.")
    except Exception as e:
        print(ctime(), f"Error adding image to Redis: {e}")
        await message.answer("Failed to add the image to the Redis index.")


@dp.message(Command("predict"))
async def predict_image_label(message: types.Message):
    """
    Predict the label for a given image based on existing samples in Redis.
    """
    if not message.reply_to_message or not (
        message.reply_to_message.photo or message.reply_to_message.document
    ):
        await message.answer(
            "Please reply to an image (photo or document) to predict its label."
        )
        return

    try:
        file_id = (
            message.reply_to_message.photo[-1].file_id
            if message.reply_to_message.photo
            else message.reply_to_message.document.file_id
        )
        file = await bot.download(file_id)

        image = np.array(Image.open(BytesIO(file.read())).convert("L"))
        if image.shape != (8, 8):
            await message.answer("Image must be 8x8")
            return

        image = image.flatten()

        res = predict_label(redis_client, image.astype(np.float16))
        await message.answer(
            f"Predicted label = {res['label']} with score = {res['score']}"
        )
    except Exception as e:
        print(ctime(), f"Error predicting label: {e}")
        await message.answer("Failed to predict the label.")


@dp.message(Command("start"))
async def start_command(message: types.Message):
    """
    Handle the /start command. Provide a warm welcome and introduction.
    """
    welcome_text = (
        "🤖 <b>Welcome to the MNIST Classifier Bot!</b>\n\n"
        "This bot allows you to interact with an image classification model using Redis.\n\n"
        "Available commands:\n"
        "📥 <b>/init</b> - Initialize the Redis index with default MNIST data.\n"
        "🖼️ <b>/add</b> - Add a custom image to the Redis index. Reply to an image with this command and include the label.\n"
        "🔍 <b>/predict</b> - Predict the label for an image. Reply to an image with this command.\n"
        "ℹ️ <b>/help</b> - Show this help message.\n\n"
        "Send /help anytime for more information."
    )
    await message.answer(welcome_text)


@dp.message(Command("help"))
async def help_command(message: types.Message):
    """
    Handle the /help command. Provide detailed usage information.
    """
    help_text = (
        "ℹ️ <b>Help - MNIST Classifier Bot</b>\n\n"
        "Commands:\n"
        "1️⃣ <b>/init</b>: Initialize the Redis index with MNIST data. Use this first!\n\n"
        "2️⃣ <b>/add &lt;label&gt;</b>: Add a custom image with a label to the Redis index.\n"
        "   - Reply to an image (photo or document) with this command and provide the label.\n"
        "   - Example: <code>/add 5</code>\n\n"
        "3️⃣ <b>/predict</b>: Predict the label of an image based on existing samples.\n"
        "   - Reply to an image (photo or document) with this command.\n\n"
        "Enjoy using the bot! 🚀"
    )
    await message.answer(help_text)




async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

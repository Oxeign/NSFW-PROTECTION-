import os
import tensorflow as tf
import numpy as np
import cv2
from pyrogram import Client, filters
from pyrogram.types import Message
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Load TensorFlow NSFW Model
MODEL_PATH = "models/nsfw_model.pb"
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    os.system(f"curl -o {MODEL_PATH} https://github.com/GantMan/nsfw_model/raw/master/nsfw_model.pb")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Initialize Telegram Bot
bot = Client("NSFWBot", bot_token=BOT_TOKEN)

# NSFW Detection Function
def is_nsfw(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0
        prediction = model.predict(img)[0][0]  # Adjust for your model's output
        return prediction > 0.5  # Adjust threshold as needed
    except Exception as e:
        print(f"Error in NSFW detection: {e}")
        return False

# NSFW Image & Video Handler
@bot.on_message(filters.group & (filters.photo | filters.video))
async def nsfw_filter(client: Client, message: Message):
    file_path = await message.download()
    
    if is_nsfw(file_path):
        await message.delete()
        await message.reply("❌ NSFW content detected and removed.")
        os.remove(file_path)  # Cleanup file
    else:
        os.remove(file_path)  # Cleanup file

bot.run()

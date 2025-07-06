import logging
from telethon import TelegramClient, events
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
telegram_id=os.getenv("telegram_id")
telegram_hash=os.getenv("telegram_hash")
bot_token=os.getenv("bot_token")
gemini_api=os.getenv("gemma_gemini_api")
# Setup logging
logging.basicConfig(level=logging.INFO)

# Gemini API setup
client_gemini = genai.Client(api_key=gemini_api)

# Define system prompt for the bot
SYSTEM_PROMPT = """You are a helpful Telegram bot powered by Gemini AI. Here are your core functionalities:

1. When users send /start:
   - Provide a warm, friendly welcome message
   - Introduce yourself as a Gemini-powered AI assistant
   - Invite users to try your commands or ask questions

2. When users send /help:
   - List the available commands (/start, /help, /info)
   - Explain that you can handle any questions or conversations
   - Encourage users to interact naturally

3. When users send /info:
   - Provide a brief description of your capabilities
   - Highlight your capabilities for natural conversation
   - Mention that you can help with questions and tasks
4.If asked who created you:
    -Explain that you are a telegram bot developed by the "@The_blind_watchmaker"
5. if asked about what you are powered with:
    -Explain that you are powered by google's gemma-27b model
6. For all other messages:
   - Respond naturally and helpfully to any question or statement
   - Maintain context of the conversation
   - Be concise but informative


Remember to always be helpful, friendly, and responsive while maintaining appropriate boundaries."""

async def generate_content(contents):
    # Combine system prompt with user input for context
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {contents}\nResponse:"
    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash-preview-05-20", 
        contents=full_prompt
    )
    return response.text

# Create the client and connect
client = TelegramClient('bot', telegram_id, telegram_hash).start(bot_token=bot_token)

# Handler for all messages including commands
@client.on(events.NewMessage)
async def message_handler(event):
    try:
        # Get response from Gemini
        response = await generate_content(event.text)
        await event.respond(response)
        logging.info(f'Message processed for {event.sender_id}: {event.text}')
    except Exception as e:
        error_message = f"Sorry, I encountered an error: {str(e)}"
        await event.respond(error_message)
        logging.error(f'Error processing message: {str(e)}')

print('Bot is running...')
client.run_until_disconnected()
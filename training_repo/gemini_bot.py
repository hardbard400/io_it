# gemini_bot.py

import logging
import os
from collections import defaultdict
from telethon import TelegramClient, events
from dotenv import load_dotenv

# Import the refactored modules
import pdf_processor
import pinecone_manager

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load credentials from .env file
API_ID = os.getenv("telegram_id")
API_HASH = os.getenv("telegram_hash")
BOT_TOKEN = os.getenv("embedding_bot_token")

# In-memory storage for user state and chat history
# Format: {chat_id: {'state': 'chatting'/'awaiting_pdf', 'history': [], 'pinecone_index': '...'}}
user_data = defaultdict(lambda: {'state': 'chatting', 'history': [], 'pinecone_index': None})

# Initialize Telegram Client
client = TelegramClient('bot_session', API_ID, API_HASH).start(bot_token=BOT_TOKEN)

# --- Command Handlers ---

@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    chat_id = event.chat_id
    user_data[chat_id]['state'] = 'chatting'
    user_data[chat_id]['history'] = []
    user_data[chat_id]['pinecone_index'] = None
    
    welcome_message = (
        "Hello! I am a Gemini-powered bot.\n\n"
        "You can chat with me, or you can upload a PDF document for me to analyze.\n\n"
        "Use /upload to start the document analysis process."
    )
    await event.respond(welcome_message)
    logging.info(f"Started new session for chat_id: {chat_id}")

@client.on(events.NewMessage(pattern='/upload'))
async def upload_command(event):
    chat_id = event.chat_id
    user_data[chat_id]['state'] = 'awaiting_pdf'
    await event.respond("Please send the PDF file you would like me to analyze.")
    logging.info(f"Chat ID {chat_id} is now awaiting PDF.")

# --- Main Message Handler ---

@client.on(events.NewMessage)
async def message_handler(event):
    # Ignore commands in the general handler
    if event.text.startswith('/'):
        return

    chat_id = event.chat_id
    current_state = user_data[chat_id]['state']
    
    # --- PDF Processing Logic ---
    if event.document and event.document.mime_type == 'application/pdf':
        if current_state == 'awaiting_pdf':
            await event.respond("✅ PDF received. Processing and embedding... This may take a few moments.")
            
            # Download the PDF
            file_path = await client.download_media(event.message.document)
            logging.info(f"Downloaded PDF to {file_path} for chat_id {chat_id}")
            
            try:
                # 1. Process PDF to JSON
                records = pdf_processor.process_pdf_to_json(file_path)
                
                if records:
                    # Pinecone index name specific to the user
                    index_name = f"user-{chat_id}"
                    user_data[chat_id]['pinecone_index'] = index_name
                    
                    # 2. Upsert data to Pinecone
                    success = pinecone_manager.upsert_to_pinecone(index_name, records)
                    
                    if success:
                        await event.respond("✅ Document processing complete! You can now ask me questions about its contents.")
                        user_data[chat_id]['state'] = 'querying_document'
                        user_data[chat_id]['history'] = [] # Reset history for the new context
                    else:
                        await event.respond("❌ There was an error embedding the document. Please try again.")
                        user_data[chat_id]['state'] = 'chatting'
                else:
                    await event.respond("❌ I couldn't extract any content from the PDF. Please ensure it contains text and try again.")
                    user_data[chat_id]['state'] = 'chatting'

            except Exception as e:
                logging.error(f"Error during PDF processing for chat_id {chat_id}: {e}")
                await event.respond("❌ An unexpected error occurred during processing.")
            finally:
                # Clean up the downloaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Removed temporary file: {file_path}")
        else:
            await event.respond("If you want me to analyze this PDF, please use the /upload command first.")
        return

    # --- Querying Logic ---
    if current_state == 'querying_document':
        async with event.client.action(event.chat_id, 'typing'):
            user_query = event.text
            index_name = user_data[chat_id]['pinecone_index']
            history = user_data[chat_id]['history']
            
            # 3. Query Pinecone and get summarized answer
            response_text = pinecone_manager.query_pinecone(index_name, user_query, history)
            
            await event.respond(response_text)
            
            # Update chat history
            user_data[chat_id]['history'].append({'role': 'user', 'content': user_query})
            user_data[chat_id]['history'].append({'role': 'assistant', 'content': response_text})

    # --- Default Chat Logic (if no document is being queried) ---
    elif current_state == 'chatting':
        await event.respond("I'm in general chat mode. To ask questions about a document, please use /upload first.")
        # Here you could add a general Gemini chat call if desired
    
# --- Main Execution ---

async def main():
    """Main function to start the bot."""
    logging.info("Bot is starting...")
    await client.run_until_disconnected()
    logging.info("Bot has stopped.")

if __name__ == '__main__':
    client.loop.run_until_complete(main())
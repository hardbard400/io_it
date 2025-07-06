import logging
import os
import json
import asyncio # <-- Import asyncio to handle blocking tasks
from telethon import TelegramClient, events
from google import genai as gemini_client  # <-- Renamed to avoid conflict
import google.generativeai as genai_file_processing # <-- The SDK for file processing
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
telegram_id = os.getenv("telegram_id")
telegram_hash = os.getenv("telegram_hash")
bot_token = os.getenv("attemptZeroBot_token")
gemini_api = os.getenv("gemma_gemini_api")

# Setup logging
logging.basicConfig(level=logging.INFO)

# --- Gemini API Setup for Chat ---
client_gemini = gemini_client.Client(api_key=gemini_api)

# --- Gemini API Setup for File Processing ---
genai_file_processing.configure(api_key=gemini_api)

# --- System Prompt for Chatbot ---
SYSTEM_PROMPT = """You are a helpful Telegram bot powered by Gemini AI...""" # Your full system prompt here

# --- PDF Processing Function (from your second script) ---
# Note: I've made it async and added a parameter to return the filename
async def gemini_segment_sentences(pdf_file_path: str):
    """
    Processes an uploaded PDF, generates structured JSON, and saves it to a file.
    Returns the path of the generated JSON file.
    """
    try:
        print(f"Uploading file to Gemini: {pdf_file_path}")
        pdf_file = genai_file_processing.upload_file(path=pdf_file_path, display_name=os.path.basename(pdf_file_path))
        print(f"Completed upload: {pdf_file.name}")

        model = genai_file_processing.GenerativeModel(model_name="models/gemini-1.5-flash")
        
        prompt = """
        Your task is to process the uploaded PDF document and convert its content into a structured JSON format...
        """ # Your full PDF processing prompt here

        print("Generating content from the PDF...")
        generation_config = genai_file_processing.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content([pdf_file, prompt], generation_config=generation_config)

        print("Content generation complete. Saving to file...")
        
        # Define a unique output filename
        output_filename = f"processed_{os.path.basename(pdf_file_path)}.json"
        
        records_data = json.loads(response.text)
        with open(output_filename, "w") as f:
            json.dump(records_data, f, indent=4)

        print(f"Successfully saved records to {output_filename}")
        # Clean up the uploaded file from Gemini servers
        genai_file_processing.delete_file(pdf_file.name)
        print(f"Cleaned up file {pdf_file.name} from Gemini server.")
        return output_filename

    except json.JSONDecodeError:
        print(f"Error: The model did not return valid JSON. The response was:\n{response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during PDF processing: {e}")
        return None


# --- Telegram Bot Logic ---

# Function to generate chat content
async def generate_chat_content(contents):
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {contents}\nResponse:"
    response = client_gemini.models.generate_content(
        model="models/gemini-1.5-flash-latest", # Using a standard model name
        contents=full_prompt
    )
    return response.text

# Create the client and connect
# Make sure you have deleted your old 'bot.session' file before running this
client = TelegramClient('bot', telegram_id, telegram_hash)

# Handler for commands (Unchanged)
@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    # ... your start logic ...
    await event.respond("Hello! I'm a Gemini-powered bot. Ask me anything or send me a PDF to process!")

@client.on(events.NewMessage(pattern='/help'))
async def help_command(event):
    # ... your help logic ...
     await event.respond("Commands: /start, /help, /info. Just send me a message for a chat, or upload a PDF file for processing.")


@client.on(events.NewMessage(pattern='/info'))
async def info(event):
    # ... your info logic ...
    await event.respond("I am powered by Google's Gemini models. I can chat with you or process PDF documents to extract structured data.")


# --- MODIFIED: General message handler to process text OR files ---
@client.on(events.NewMessage(incoming=True))
async def message_handler(event):
    # Ignore commands in this general handler
    if event.text and event.text.startswith('/'):
        return

    # --- 1. NEW: Check if the message contains a document ---
    if event.document and event.document.mime_type == 'application/pdf':
        try:
            # Let the user know you've received the file
            await event.respond("✅ PDF received! Starting to process it now. This may take a few moments...")
            
            # --- 2. Download the file from Telegram ---
            # The downloaded file will be named after the original filename
            download_path = await event.download_media()
            logging.info(f"PDF downloaded to: {download_path}")

            # --- 3. Run the blocking PDF processing function in a separate thread ---
            # This prevents the bot from freezing while waiting for Gemini
            json_output_file = await asyncio.to_thread(gemini_segment_sentences, download_path)

            # --- 4. Send the result back to the user ---
            if json_output_file:
                await client.send_file(
                    event.chat_id,
                    json_output_file,
                    caption="🎉 Processing complete! Here is your structured JSON file."
                )
                # Clean up the local files
                os.remove(download_path)
                os.remove(json_output_file)
            else:
                await event.respond("❌ Sorry, something went wrong while processing your PDF.")

        except Exception as e:
            error_message = f"Sorry, I encountered an error handling your file: {str(e)}"
            await event.respond(error_message)
            logging.error(f'Error processing file: {str(e)}')

    # --- Handle regular text messages (your original logic) ---
    elif event.text:
        try:
            response = await generate_chat_content(event.text)
            await event.respond(response)
            logging.info(f'Message processed by Gemini for {event.sender_id}: {event.text}')
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            await event.respond(error_message)
            logging.error(f'Error processing message: {str(e)}')


# --- Start the Client ---
async def main():
    await client.start(bot_token=bot_token)
    print("Bot started...")
    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())
    
import logging
import os
import json
import random
import string
import time
from dotenv import load_dotenv
from telethon import TelegramClient, events
import google.generativeai as genai
from pinecone import Pinecone

# --- Part 0: Initial Setup and Configuration ---

# Load all environment variables from .env file
load_dotenv()
telegram_id = os.getenv("telegram_id")
telegram_hash = os.getenv("telegram_hash")
bot_token = os.getenv("embedding_bot_token")
gemini_api_key = os.getenv("gemma_gemini_api")
pinecone_api = os.getenv("pinecone_api")

# Configure APIs
logging.basicConfig(level=logging.INFO)
genai.configure(api_key=gemini_api_key)
pc = Pinecone(api_key=pinecone_api)

# In-memory storage for user session context (Pinecone index/namespace)
user_session_context = {}


# --- Part 1: Logic from gemini_record.py ---

def convert_pdf_to_json_records(pdf_file_path: str):
    """
    Processes the uploaded PDF document and converts its content into a
    structured JSON format, saving it to 'records.json'.
    """
    try:
        logging.info(f"Uploading file to Gemini: {pdf_file_path}")
        pdf_file = genai.upload_file(path=pdf_file_path, display_name=os.path.basename(pdf_file_path))
        logging.info(f"Completed upload: {pdf_file.name}")

        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        
        prompt = """
        Your task is to process the uploaded PDF document and convert its content into a structured JSON format for embedding purposes.

        Follow these instructions precisely:

        1.  For each of the 9 pages in the document, you must generate exactly 60 concise, self-contained statements. This will result in a total of 540 statements (9 pages * 60 statements/page).
        2.  Each statement should be a separate record in a JSON array.
        3.  Each record in the JSON array must have the following three fields:
            * `_id`: A unique identifier for the record. Start with "rec1" and increment for each subsequent record (e.g., "rec2", "rec3", ... "rec540").
            * `chunk_text`: The statement you generated from the PDF's content. This should be a single, complete sentence.
            * `category`: This should be a single, consistent category for all records. Use "document_analysis" for this field.
        4.  The final output must be a single JSON object containing a list named "records".

        Here is an example of the required JSON format:
        {
          "records": [
            {
              "_id": "rec1",
              "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.",
              "category": "document_analysis"
            },
            {
              "_id": "rec2",
              "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.",
              "category": "document_analysis"
            }
          ]
        }

        Please begin processing the document and generate the complete JSON output with all 540 records.
        """
        logging.info("Generating structured JSON content from the PDF...")
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content([pdf_file, prompt], generation_config=generation_config)
        
        output_filename = "records.json"
        with open(output_filename, "w") as f:
            f.write(response.text)
        
        logging.info(f"Successfully saved records to {output_filename}")
        return output_filename
    except Exception as e:
        logging.error(f"An error occurred in convert_pdf_to_json_records: {e}")
        return None


# --- Part 2: Logic for Pinecone Interaction ---

def generate_random_string(length=14):
    """Generates a random string for index and namespace names."""
    characters = string.ascii_lowercase + string.digits + "-"
    return ''.join(random.choices(characters, k=length))

def embed_json_and_upsert(json_file_path):
    """
    Creates a Pinecone index using modern methods and upserts the records from the JSON file.
    """
    try:
        index_name = generate_random_string()
        namespace_name = generate_random_string()
        
        # This is the modern, official way to create an index with the new Pinecone library.
        # The dimension 768 is compatible with the 'models/text-embedding-004' model used below.
        if index_name not in pc.list_indexes().get('names', []):
            logging.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=768, 
                metric='cosine',
                spec={
                    'serverless': {
                        'cloud': 'aws',
                        'region': 'us-east-1'
                    }
                }
            )

        with open(json_file_path, 'r') as f:
            records_data = json.load(f)
            # The modern approach requires manually embedding the text before upserting.
            # We use a compatible Google model since we are using the 'genai' library.
            model_id = "models/text-embedding-004"
            
            for record in records_data['records']:
                record['values'] = genai.embed_content(model=model_id, content=record['chunk_text'])['embedding']
            
            vectors_to_upsert = [{'id': r['_id'], 'values': r['values'], 'metadata': {'chunk_text': r['chunk_text'], 'category': r['category']}} for r in records_data['records']]


        dense_index = pc.Index(index_name)
        
        # Batching remains to handle the upsert limit.
        batch_size = 96
        logging.info(f"Upserting records to index '{index_name}' in namespace '{namespace_name}' in batches of {batch_size}...")
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            dense_index.upsert(vectors=batch, namespace=namespace_name)
            logging.info(f"Upserted batch {i//batch_size + 1}")

        time.sleep(10)
        
        logging.info("Upsert complete.")
        return index_name, namespace_name

    except Exception as e:
        logging.error(f"An error occurred in embed_json_and_upsert: {e}")
        return None, None

def query_pinecone_and_summarize(index_name, namespace_name, query):
    """
    Queries the specified Pinecone index and summarizes the result.
    """
    try:
        dense_index = pc.Index(index_name)
        gemini_client = genai.GenerativeModel(model_name="gemini-2.5-flash")
        
        # The query must also be embedded into a vector to perform the search.
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )["embedding"]

        logging.info(f"Searching index '{index_name}' with query: '{query}'")
        search_results = dense_index.query(
            namespace=namespace_name,
            vector=query_embedding,
            top_k=20,
            include_metadata=True
        )

        hit_texts = [hit['metadata']['chunk_text'] for hit in search_results['matches']]
        combined_text = "\n".join(f"- {t}" for t in hit_texts)
        
        system_prompt = (
            "this is the initial query :{}, and this is the  collection of close answers:{}"
            "if you find that the answers you get dont match with the question asked, just answer on your own"
            "Summarize the following search results in a concise paragraph:\n"
            "if the contents of {} dont contain sufficient answers for {} then answer with, 'not enough data for satisfactory answer'"
            "Only summarize the parts relevant, get straight to the summarization, no need to refer to the structure"
            "answer in a maximum of 3 sentences"
            "make no mention of the search process or the findings process".format(query, combined_text, query, combined_text)
        )
        response = gemini_client.generate_content(system_prompt)
        return response.text

    except Exception as e:
        logging.error(f"An error occurred in query_pinecone_and_summarize: {e}")
        return "Sorry, I encountered an error while trying to answer your question."

# --- Part 3: The Main Telegram Bot Orchestrator ---

client = TelegramClient('bot_session', telegram_id, telegram_hash).start(bot_token=bot_token)

@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    await event.respond("Hello! I am a bot ready to process your documents.\n\nPlease send me a PDF file. I will embed its contents, and then you can ask me questions about it.")

@client.on(events.NewMessage(pattern='/help'))
async def help_command(event):
    await event.respond("1. Send a PDF file to start.\n2. Wait for the confirmation that the file is processed.\n3. Ask questions about the document's content.\n4. Use /clear to start over with a new document.")

@client.on(events.NewMessage(pattern='/clear'))
async def clear_context(event):
    chat_id = event.chat_id
    if chat_id in user_session_context:
        try:
            index_to_delete = user_session_context[chat_id]['index_name']
            if index_to_delete in pc.list_indexes().get('names', []):
                pc.delete_index(index_to_delete)
                logging.info(f"Deleted Pinecone index: {index_to_delete}")
        except Exception as e:
            logging.error(f"Could not delete index: {e}")
        
        del user_session_context[chat_id]
        await event.respond("Your session has been cleared. I am ready for a new PDF.")
    else:
        await event.respond("There is no active session to clear.")


@client.on(events.NewMessage)
async def message_handler(event):
    chat_id = event.chat_id
    
    if event.document and event.document.mime_type == 'application/pdf':
        await event.respond("PDF received. Starting the process, this may take a few moments...")
        
        file_path = await client.download_media(event.message)
        logging.info(f"PDF downloaded to: {file_path}")

        json_path = convert_pdf_to_json_records(file_path)
        
        if not json_path:
            await event.respond("Sorry, I failed to process the PDF into JSON records.")
            return

        await event.respond("PDF content extracted. Now embedding the data into Pinecone...")

        index_name, namespace_name = embed_json_and_upsert(json_path)
        
        if not index_name:
            await event.respond("Sorry, I failed to embed the document's data.")
            return

        user_session_context[chat_id] = {
            "index_name": index_name,
            "namespace_name": namespace_name
        }
        
        await event.respond("Your document has been successfully processed and embedded! You can now ask me questions about it.")
        
        os.remove(file_path)
        os.remove(json_path)
        return

    if event.text and not event.text.startswith('/'):
        if chat_id in user_session_context:
            context = user_session_context[chat_id]
            query = event.text
            
            await event.respond("Searching the document for an answer...")
            
            summary = query_pinecone_and_summarize(
                context["index_name"],
                context["namespace_name"],
                query
            )
            await event.respond(summary)
            
        else:
            await event.respond("Please send me a PDF document first before asking questions.")


# --- Main execution block ---
if __name__ == '__main__':
    logging.info("Bot is starting up...")
    client.run_until_disconnected()
    logging.info("Bot has stopped.")
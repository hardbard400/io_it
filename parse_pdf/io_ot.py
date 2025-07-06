# gemini_qa_bot.py

import os
import logging
from dotenv import load_dotenv
from telethon import TelegramClient, events
from pinecone import Pinecone
import google.generativeai as genai

# --- 1. SETUP AND INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Client Initialization ---
# Load credentials from .env file
API_ID = os.getenv("telegram_id")
API_HASH = os.getenv("telegram_hash")
BOT_TOKEN = os.getenv("attemptZeroBot_token")
PINECONE_API_KEY = os.getenv("pinecone_api")
GEMINI_API_KEY = os.getenv("gemma_gemini_api")

# Initialize all clients
client = TelegramClient('bot_session', API_ID, API_HASH).start(bot_token=BOT_TOKEN)
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# --- Pinecone and Gemini Model Setup ---
model = genai.GenerativeModel('gemini-2.5-flash')
index_names = ["biology-grade-12", "biology-grade-11"]
logging.info(f"Connected to Pinecone for indices: {index_names}")
logging.info("Gemini Model 'gemini-2.5-flash' initialized.")


# --- 2. QUERY AND ANSWER LOGIC ---

async def refine_query(user_query: str) -> str:
    """
    **NEW FUNCTION**
    Uses the AI model to refine the user's conversational query into a 
    keyword-focused search query for the vector database.
    """
    system_prompt = f"""
    You are an expert at query optimization. Your task is to take the following user's question and distill it into a concise, keyword-focused search query.
    Focus only on the core scientific or technical terms. Do not provide an answer or any explanation.

    User Question: "{user_query}"

    Refined Search Query:
    """
    try:
        response = await model.generate_content_async(system_prompt)
        refined_query = response.text.strip()
        logging.info(f"Original query: '{user_query}' | Refined to: '{refined_query}'")
        return refined_query
    except Exception as e:
        logging.error(f"Error during query refinement: {e}")
        # Fallback to the original query if refinement fails
        return user_query

async def generate_final_answer(original_query: str, contents: str) -> str:
    """
    **MODIFIED**
    Generates the final answer based on the ORIGINAL query and the retrieved context.
    The function is now asynchronous.
    """
    system_prompt = f"""
    You are a specialized AI assistant. Your sole purpose is to answer the user's query based exclusively on the provided search results. You must adhere to the following instructions without deviation.

    **Instructions:**

    1.  **Analyze the User's Original Query:** The user wants to know: "{original_query}"

    2.  **Review the Provided Context:** You are given the following search results.
        ```context
        {contents}
        ```

    3.  **Synthesize the Answer:**
        * Formulate a descriptive answer to the user's original query using *only* the information found in the `TEXT_CONTENT` of the provided results.
        * Do not invent, infer, or use any information outside of the provided context.
        * You are here to answer questions, so anwer each answer according to the format its proided in. i.e "multiple alternatives" or "fill in the blank space or "short answer questions. but also provide the explanations for your answers."
        * If you cannot find a suitable answer for "{original_query}", reply with "I'm sorry, there isn't a suitable answer to your question in the book."

    4.  **Cite Your Sources:**
        * After the answer, list the sources you used.
        * For each source, include its `ID`, `SCORE`, `TEXT_HEADER`, and `PAGE_NUMBER`.
        * Format each citation exactly as: `Source: \n ID: [ID], SCORE: [SCORE], HEADER: [TEXT_HEADER], PAGE_NUMBER: [PAGE_NUMBER]`

    **Output Mandate:**
    * Your entire output must consist of two parts ONLY: the synthesized answer first, followed by the list of source citations.
    * DO NOT add any introductory phrases, greetings, or concluding remarks.
    """
    try:
        response = await model.generate_content_async(system_prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error during final answer generation: {e}")
        return "There was an error generating the final answer."


async def process_query(user_query: str):
    """
    **MODIFIED**
    Orchestrates the new workflow:
    1. Refines the user query.
    2. Searches Pinecone using the refined query.
    3. Generates the final answer using the original query and search results.
    """
    # 1. First, refine the user's query for better search results.
    refined_search_query = await refine_query(user_query)

    # 2. Search all specified namespaces and indices using the refined query.
    namespaces = ["key_words", "general_text", "tables"]
    all_contexts = []
    
    for ind in index_names:
        try:
            dense_index = pc.Index(ind)
            for ns in namespaces:
                # Use the refined query for the database search
                results = dense_index.search(
                    namespace=ns,
                    query={
                        "top_k": 8,
                        "inputs": {
                            'text': refined_search_query
                        }
                    }
                )
                formatted_results = "\n".join(
                    f"ID: {hit.get('_id', 'N/A')} | SCORE: {round(hit.get('_score', 0), 2)} | PAGE_NUMBER: {hit.get('fields', {}).get('page_number', 'N/A')}\n"
                    f"TEXT_HEADER: {hit.get('fields', {}).get('topic', 'N/A')}\n"
                    f"TEXT_CONTENT: {hit.get('fields', {}).get('chunk_text', 'N/A')}\n\n"
                    for hit in results.get('result', {}).get('hits', [])
                )
                if formatted_results:
                    all_contexts.append(formatted_results)
        except Exception as e:
            logging.error(f"Error searching Pinecone index '{ind}', namespace '{ns}': {e}")
    
    full_context = "\n".join(all_contexts)

    if not full_context.strip():
        logging.warning("No context found from Pinecone search.")
        return "I'm sorry, I couldn't find any relevant information in the book to answer your question."

    # 3. Generate the final answer using the original query and the retrieved context.
    final_answer = await generate_final_answer(user_query, full_context)
    return final_answer


# --- 3. TELEGRAM BOT HANDLERS ---

@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    """Handles the /start command."""
    welcome_message = (
        "Hello! I am a Q&A bot for the Grade 11 & 12 Biology curriculum.\n\n"
        "Please ask me a question, and I will find the answer for you from the textbook."
    )
    await event.respond(welcome_message)
    logging.info(f"Started new session for chat_id: {event.chat_id}")


@client.on(events.NewMessage)
async def message_handler(event):
    """Handles all non-command text messages."""
    if event.text.startswith('/'):
        return

    user_query = event.text
    chat_id = event.chat_id
    logging.info(f"Received query from chat_id {chat_id}: '{user_query}'")

    async with client.action(chat_id, 'typing'):
        try:
            # The call to process_query now triggers the new, refined workflow
            response_text = await process_query(user_query)
            await event.respond(response_text)
            logging.info(f"Successfully sent response to chat_id {chat_id}")
        except Exception as e:
            logging.error(f"An error occurred in message_handler for chat_id {chat_id}: {e}")
            await event.respond("I'm sorry, an unexpected error occurred. Please try again later.")


# --- 4. MAIN EXECUTION BLOCK ---

async def main():
    """Main function to run the bot."""
    logging.info("Bot is starting up...")
    await client.run_until_disconnected()
    logging.info("Bot has stopped.")

if __name__ == '__main__':
    client.loop.run_until_complete(main())
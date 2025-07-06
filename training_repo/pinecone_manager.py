# pinecone_manager.py

import os
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai

def upsert_to_pinecone(index_name: str, records: list):
    """
    Initializes a Pinecone index (creating it if it doesn't exist) and
    upserts records into it.

    Args:
        index_name: The name of the Pinecone index.
        records: A list of records to upsert.

    Returns:
        True if successful, False otherwise.
    """
    load_dotenv()
    pinecone_api = os.getenv("PINECONE_API_KEY")
    if not pinecone_api or not records:
        print("Error: Pinecone API key or records are missing.")
        return False

    pc = Pinecone(api_key=pinecone_api)

    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768, # Dimensions for text-embedding-ada-002
            metric='cosine',
            spec={
                'serverless': {
                    'cloud': 'aws',
                    'region': 'us-east-1'
                }
            }
        )
    
    index = pc.Index(index_name)
    print(f"Upserting {len(records)} records to index '{index_name}'...")
    
    # Use Gemini for embedding
    model = 'models/text-embedding-004'
    
    def embed(texts):
        return genai.embed_content(model=model, content=texts, task_type="retrieval_document")['embedding']

    # Prepare vectors for upsert
    vectors_to_upsert = []
    for i, record in enumerate(records):
        vector = embed([record['chunk_text']])[0]
        vectors_to_upsert.append({
            "id": f"rec-{i}",
            "values": vector,
            "metadata": record
        })

    # Upsert in batches to avoid overwhelming the API
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//batch_size + 1}")

    print("Upsert complete.")
    return True

def query_pinecone(index_name: str, user_query: str, chat_history: list):
    """
    Queries a Pinecone index, combines the results with chat history,
    and generates a summarized answer using Gemini.

    Args:
        index_name: The name of the Pinecone index to query.
        user_query: The user's question.
        chat_history: A list of previous conversation turns.

    Returns:
        A summarized, conversational string response.
    """
    load_dotenv()
    pinecone_api = os.getenv("PINECONE_API_KEY")
    gemini_api = os.getenv("GEMINI_API_KEY")

    if not all([pinecone_api, gemini_api, user_query]):
        return "Configuration error or missing query."

    pc = Pinecone(api_key=pinecone_api)
    genai.configure(api_key=gemini_api)

    if index_name not in pc.list_indexes().names():
        return "It seems no document has been uploaded yet. Please use /upload to add a PDF."

    index = pc.Index(index_name)
    gemini_client = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

    # Embed the user query
    model = 'models/text-embedding-004'
    query_embedding = genai.embed_content(
        model=model,
        content=user_query,
        task_type="retrieval_query"
    )['embedding']
    
    # Query Pinecone
    query_results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    # Consolidate context from search results
    context_texts = [match['metadata'].get('chunk_text', '') for match in query_results['matches']]
    combined_context = "\n".join(f"- {text}" for text in context_texts)

    # Format chat history for the prompt
    formatted_history = "\n".join([f"{turn['role']}: {turn['content']}" for turn in chat_history])

    system_prompt = f"""
    You are an intelligent assistant. Your task is to answer a user's query based on the provided context from a document they uploaded and the history of your conversation.

    Conversation History:
    {formatted_history}

    Context from the document:
    {combined_context}

    User's latest query: "{user_query}"

    Instructions:
    1.  Synthesize an answer ONLY from the provided "Context from the document".
    2.  If the context does not contain the answer, state that the document does not provide enough information to answer the question. Do not use external knowledge.
    3.  Incorporate the "Conversation History" to understand follow-up questions and maintain a natural, conversational flow.
    4.  Keep your answer concise and directly address the user's query.
    """
    
    response = gemini_client.generate_content(system_prompt)
    return response.text
import json
from dotenv import load_dotenv
import os
import time

# --- 1. Load Environment Variables and Data ---
load_dotenv()
with open("/workspaces/io_it/parse_pdf/Grade_9_Biology_keyword_definitions.json", "r") as file:
    records = json.load(file)

# --- 2. Prepare Records ---
# Pinecone requires each record to have an 'id' field that is a string.
# This loop ensures all IDs are strings before we begin batching.
# for record in records:
#     record['_id'] = str(record['_id'])

# --- 3. Initialize Pinecone ---
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("pinecone_api"))

# --- 4. Create Index if it Doesn't Exist ---
index_name = "biology"
if not pc.has_index(index_name):
    print(f"Index '{index_name}' not found. Creating a new one...")
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )
    # Add a small delay to ensure the index is ready
    time.sleep(5)
else:
    print(f"Found existing index '{index_name}'.")

# Target the index
dense_index = pc.Index(index_name)

# --- 5. Upsert Records in Batches (The New Structure) ---
batch_size = 96  # Set the batch size as specified by the error message

print(f"Total records to upsert: {len(records)}")
print(f"Upserting in batches of {batch_size}...")

# This loop iterates through the 'records' list in steps of 'batch_size'
for i in range(0, len(records), batch_size):
    # Slice the list to get the current batch
    batch = records[i:i + batch_size]
    
    # Get the current batch number for logging
    batch_num = (i // batch_size) + 1
    
    print(f"--> Upserting batch {batch_num} with {len(batch)} records...")
    
    # Upsert the current batch into the 'keywords' namespace
    dense_index.upsert_records(namespace="Grade-9-biology-keyword-definitions", records=batch)

print("\nAll batches have been successfully upserted.")

# --- 6. Verify the Upload ---
# It's good practice to wait a moment for the index to update
print("Waiting for index to update...")
time.sleep(10)

# View stats for the index to confirm the record count
stats = dense_index.describe_index_stats()
print("\n--- Index Stats ---")
print(stats)
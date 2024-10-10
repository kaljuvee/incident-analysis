import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables
INPUT_DIR = "data/small"
MODEL_NAME = "gpt-4o"
NUM_CHUNKS = 10

def load_documents(input_dir):
    documents = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), 'r') as file:
                documents.append(file.read())
    return "\n\n".join(documents)  # Combine all documents into one large string

def split_text(text, num_chunks):
    # Simple text splitter that splits the text into roughly equal chunks
    total_length = len(text)
    chunk_size = total_length // num_chunks
    chunks = []
    
    for i in range(0, total_length, chunk_size):
        chunk = text[i:i+chunk_size]
        # Adjust chunk boundary to the nearest newline
        if i + chunk_size < total_length:
            next_newline = text.find('\n', i + chunk_size)
            if next_newline != -1:
                chunk = text[i:next_newline]
        chunks.append(chunk)
    
    return chunks

def query_chunk(query, chunk):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that identifies and counts incidents based on the given context."},
        {"role": "user", "content": f"Context: {chunk}\n\nQuestion: {query}\n\nPlease identify all the {query} incidents in the given context and provide a count of how many there are. Only return the number, nothing else."}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=50,
        n=1,
        temperature=0,
    )

    return int(response.choices[0].message.content.strip())

def main():
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found in environment variables.")
        logging.error("Please make sure you have a .env file with your OpenAI API key.")
        return

    # Load documents
    logging.info("Loading documents")
    document = load_documents(INPUT_DIR)
    logging.info(f"Loaded and combined documents")

    # Split the document into chunks
    chunks = split_text(document, NUM_CHUNKS)
    logging.info(f"Split document into {len(chunks)} chunks")

    # Hardcoded incident type to query
    incident_type = "beehive"

    # Query each chunk and sum up the counts
    total_count = 0
    for i, chunk in enumerate(chunks, 1):
        logging.info(f"Querying chunk {i}/{len(chunks)} for incident type: '{incident_type}' using model: {MODEL_NAME}")
        count = query_chunk(incident_type, chunk)
        total_count += count
        logging.info(f"Chunk {i} count: {count}")

    print(f"\nTotal {incident_type} incidents found: {total_count}")

if __name__ == "__main__":
    main()
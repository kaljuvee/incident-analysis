import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MODEL_NAME = "gpt-4"
DATA_DIR = "data/small"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to read documents from the input folder
def read_documents(input_folder):
    logging.info(f"Reading documents from {input_folder}")
    combined_content = ""
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            logging.info(f"Reading file: {filename}")
            with open(os.path.join(input_folder, filename), 'r') as file:
                combined_content += file.read().strip() + "\n\n"
    logging.info(f"Total characters read: {len(combined_content)}")
    return combined_content

# Function to create embeddings and index
def create_embeddings_index(content):
    logging.info("Creating embeddings and index")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    logging.info(f"Created index with {len(chunks)} chunks")
    return vector_store

# Function to count incidents using GPT model
def count_incidents(vector_store, query_incident):
    logging.info(f"Counting incidents for query: '{query_incident}'")
    
    # Retrieve relevant chunks
    relevant_chunks = vector_store.similarity_search(query_incident, k=5)
    combined_chunks = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
    prompt = f"""Analyze the following incident report excerpts and count how many times the incident type '{query_incident}' is mentioned. Provide the response as a single number.

Incident report excerpts:
{combined_chunks}

Respond with only the count as a number, with no additional text or explanations. If no incidents of the specified type are found, respond with '0'."""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in analyzing incident reports. Respond only with the incident count as a number."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10
    )
    
    count = int(response.choices[0].message.content.strip())
    logging.info(f"Incident count: {count}")
    return count

def main():
    logging.info("Starting the analysis process")
    
    # Read and combine documents
    combined_content = read_documents(DATA_DIR)

    # Create embeddings and index
    vector_store = create_embeddings_index(combined_content)

    # Set the query incident
    query_incident = "fire"
    logging.info(f"Query incident set to: '{query_incident}'")

    # Count incidents
    total_count = count_incidents(vector_store, query_incident)

    # Print total count
    logging.info(f"Analysis complete. Total count for '{query_incident}' incidents: {total_count}")
    print(f"Total count for '{query_incident}' incidents: {total_count}")

if __name__ == "__main__":
    main()
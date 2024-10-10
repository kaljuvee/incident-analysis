import os
import json
import logging
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MODEL_NAME = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"
DATA_DIR = "data/small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to read documents from the input folder
def read_documents(input_folder):
    documents = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(input_folder, filename), 'r') as file:
                content = file.read().strip()
                documents.append(content)
    logging.info(f"Read {len(documents)} documents from {input_folder}")
    return documents

def create_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def extract_incidents(content):
    prompt = f"""Analyze the following incident report and identify all unique types of incidents mentioned. Then, count how many times each incident type appears. Provide the response in this exact format:
incident: <incident type>, count: <incident count>

Incident report:
{content}

Provide only the list of incidents and counts, with no additional text or explanations. If no incidents are found, respond with 'incident: none, count: 0'."""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in analyzing incident reports. Respond only with the incident type and count."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip().lower()

def parse_incidents(incident_string):
    incidents = {}
    for line in incident_string.split('\n'):
        if ':' in line:
            incident, count = line.split(',')
            incident_type = incident.split(':')[1].strip()
            count = int(count.split(':')[1].strip())
            incidents[incident_type] = count
    return incidents

def analyze_documents(documents):
    discovered_incidents = Counter()
    all_chunks = []
    all_embeddings = []

    logging.info("Creating chunks and embeddings")
    for document in documents:
        chunks = create_chunks(document)
        all_chunks.extend(chunks)
        for chunk in chunks:
            embedding = get_embedding(chunk)
            all_embeddings.append(embedding)

    logging.info(f"Created {len(all_chunks)} chunks with embeddings")

    # Create a query embedding
    query_embedding = get_embedding("incident report analysis")

    # Calculate cosine similarities
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]

    # Sort chunks by similarity
    sorted_indices = np.argsort(similarities)[::-1]

    logging.info("Analyzing most relevant chunks")
    for i in sorted_indices[:min(50, len(sorted_indices))]:  # Analyze top 50 most relevant chunks
        chunk = all_chunks[i]
        incident_string = extract_incidents(chunk)
        incidents = parse_incidents(incident_string)
        discovered_incidents.update(incidents)

    logging.info("Chunk analysis completed")
    return discovered_incidents

def main():
    logging.info("Starting incident analysis process")
    
    # Read documents
    documents = read_documents(DATA_DIR)

    # Analyze documents
    discovered_incidents = analyze_documents(documents)

    # Print total counts
    logging.info("Printing total incident counts")
    print("Total incident counts:")
    for incident_type, count in discovered_incidents.items():
        print(f"{incident_type}: {count}")
    
    logging.info("Incident analysis process completed")

if __name__ == "__main__":
    main()
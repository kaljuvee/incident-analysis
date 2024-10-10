import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MODEL_NAME = "gpt-4o"
DATA_DIR = "data/small"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to read documents from the input folder
def read_documents(input_folder):
    logging.info(f"Reading documents from {input_folder}")
    documents = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            logging.info(f"Reading file: {filename}")
            with open(os.path.join(input_folder, filename), 'r') as file:
                content = file.read().strip()
                documents.append(content)
    logging.info(f"Total documents read: {len(documents)}")
    return documents

# Function to count incidents using GPT model
def count_incidents(content, query_incident):
    logging.info(f"Counting incidents for query: '{query_incident}'")
    prompt = f"""Analyze the following incident report and count how many times the incident type '{query_incident}' is mentioned. Provide the response as a single number.

Incident report:
{content}

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

def analyze_documents(documents, query_incident):
    logging.info(f"Analyzing documents for incident type: '{query_incident}'")
    total_count = 0
    for i, content in enumerate(documents, 1):
        logging.info(f"Analyzing document {i}/{len(documents)}")
        count = count_incidents(content, query_incident)
        total_count += count
    logging.info(f"Total incidents found: {total_count}")
    return total_count

def main():
    logging.info("Starting the analysis process")
    
    # Read documents
    documents = read_documents(DATA_DIR)

    # Set the query incident
    query_incident = "fire"
    logging.info(f"Query incident set to: '{query_incident}'")

    # Analyze documents
    total_count = analyze_documents(documents, query_incident)

    # Print total count
    logging.info(f"Analysis complete. Total count for '{query_incident}' incidents: {total_count}")
    print(f"Total count for '{query_incident}' incidents: {total_count}")

if __name__ == "__main__":
    main()
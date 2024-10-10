import os
import json
import logging
from collections import Counter
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
    documents = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(input_folder, filename), 'r') as file:
                content = file.read().strip()
                documents.append(content)
    logging.info(f"Read {len(documents)} documents from {input_folder}")
    return documents

# Function to extract incidents using GPT model
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
    logging.info(f"Starting analysis of {len(documents)} documents")
    
    for i, content in enumerate(documents, 1):
        incident_string = extract_incidents(content)
        incidents = parse_incidents(incident_string)
        discovered_incidents.update(incidents)
        if i % 10 == 0:
            logging.info(f"Processed {i} documents")
    logging.info("Document analysis completed")
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
import os
import json
import logging
import pandas as pd
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Directory to store files
output_dir = "data/small"
os.makedirs(output_dir, exist_ok=True)

# Function to generate a synthetic document
def generate_synthetic_document(incident_type):
    prompt = f"Generate a 2-sentence incident report for a {incident_type} in a manufacturing plant. The first sentence should describe the incident, and the second sentence should mention potential consequences."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant that generates concise incident reports for manufacturing plants."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    
    return response.choices[0].message.content.strip()

# Generate documents
def generate_documents(incident_types, output_dir, num_documents):
    incident_count = Counter()
    incident_id = 1

    for incident_type, percentage in incident_types.items():
        count = int(num_documents * percentage / 100)
        for _ in range(count):
            if sum(incident_count.values()) >= num_documents:
                break
            
            document_content = generate_synthetic_document(incident_type)
            
            # Save document to a TXT file
            file_name = f"incident_{incident_id}.txt"
            with open(os.path.join(output_dir, file_name), 'w') as file:
                file.write(document_content)
            
            incident_count[incident_type] += 1
            incident_id += 1
            
            # Log progress
            logging.info(f"Generated document {incident_id-1}/{num_documents}: {incident_type}")
        
        if sum(incident_count.values()) >= num_documents:
            break
    
    return incident_count

# Function to analyze generated documents
def analyze_documents(output_dir):
    discovered_incidents = Counter()
    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(output_dir, filename), 'r') as file:
                content = file.read()
                incident_type = extract_incident_type(content)
                discovered_incidents[incident_type] += 1
    return discovered_incidents

# Function to extract incident type from document content
def extract_incident_type(content):
    prompt = f"Extract the incident type from this report. Respond with only the incident type, nothing else:\n\n{content}"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that extracts incident types from incident reports."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=20
    )
    
    return response.choices[0].message.content.strip().lower()

# Main function
def main():
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found in environment variables.")
        logging.error("Please make sure you have a .env file with your OpenAI API key.")
        return

    # Load incident type distribution
    with open(os.path.join(output_dir, "incident_type_distribution.json"), 'r') as f:
        incident_types = json.load(f)

    # Number of documents to generate
    num_documents = 100

    logging.info(f"Starting document generation. Target: {num_documents} documents")
    incident_count = generate_documents(incident_types, output_dir, num_documents)
    logging.info(f"Document generation complete. Total generated: {sum(incident_count.values())}")
    logging.info(f"Documents saved in: {output_dir}")
    logging.info("Incident type distribution:")
    for incident_type, percentage in incident_types.items():
        logging.info(f"  {incident_type}: {percentage}%")

    # Analyze generated documents
    discovered_incidents = analyze_documents(output_dir)

    # Create DataFrame for comparison
    data = []
    for incident_type, percentage in incident_types.items():
        ground_truth_count = int(num_documents * percentage / 100)
        discovered_count = discovered_incidents.get(incident_type, 0)
        data.append({
            'incident_name_ground_truth': incident_type,
            'ground_truth_count': ground_truth_count,
            'incident_name_discovered': incident_type,
            'count_discovered': discovered_count
        })

    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    csv_file = os.path.join(output_dir, "incident_report.csv")
    df.to_csv(csv_file, index=False)
    logging.info(f"Incident report saved to: {csv_file}")

if __name__ == "__main__":
    main()

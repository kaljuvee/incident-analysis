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

# Directory to read files from
input_dir = "data/small"

def extract_incidents(content):
    prompt = f"""Analyze the following incident report and identify all unique types of incidents mentioned. Then, count how many times each incident type appears. Provide the response in this exact format:
incident: <incident type>, count: <incident count>

Incident report:
{content}

Provide only the list of incidents and counts, with no additional text or explanations. If no incidents are found, respond with 'incident: none, count: 0'."""
    
    response = client.chat.completions.create(
        model="gpt-4",
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

def analyze_documents(input_dir):
    discovered_incidents = Counter()
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), 'r') as file:
                content = file.read()
                incident_string = extract_incidents(content)
                incidents = parse_incidents(incident_string)
                discovered_incidents.update(incidents)
    return discovered_incidents

def main():
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found in environment variables.")
        logging.error("Please make sure you have a .env file with your OpenAI API key.")
        return

    # Analyze documents
    logging.info("Starting document analysis")
    discovered_incidents = analyze_documents(input_dir)
    logging.info("Document analysis complete")

    # Load incident type distribution
    with open(os.path.join(input_dir, "incident_type_distribution.json"), 'r') as f:
        incident_types = json.load(f)

    # Create DataFrame for comparison
    data = []
    total_documents = sum(discovered_incidents.values())
    for incident_type in set(list(incident_types.keys()) + list(discovered_incidents.keys())):
        ground_truth_percentage = incident_types.get(incident_type, 0)
        ground_truth_count = int(total_documents * ground_truth_percentage / 100)
        discovered_count = discovered_incidents.get(incident_type, 0)
        data.append({
            'incident_name_ground_truth': incident_type,
            'ground_truth_count': ground_truth_count,
            'incident_name_discovered': incident_type,
            'count_discovered': discovered_count
        })

    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    csv_file = os.path.join(input_dir, "incident_report_discovered.csv")
    df.to_csv(csv_file, index=False)
    logging.info(f"Incident report saved to: {csv_file}")

if __name__ == "__main__":
    main()
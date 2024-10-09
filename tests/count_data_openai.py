import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_documents(input_folder):
    documents = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(input_folder, filename), 'r') as file:
                content = file.read().strip()
                documents.append(content)
    return documents

def analyze_incidents(documents):
    incidents_text = "\n\n".join([f"Incident {i+1}: {doc}" for i, doc in enumerate(documents)])
    prompt = f"""Analyze the following incident reports:

{incidents_text}

Identify all unique types of incidents mentioned in these reports. For each identified incident type, provide the response in this exact format:
incident: <incident type>, count: <incident count>

Provide only the list of incidents and counts, with no additional text or explanations."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in analyzing incident reports. Respond only in the exact format specified in the prompt."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def parse_analysis(analysis):
    discovered_incidents = {}
    lines = analysis.split('\n')
    
    for line in lines:
        match = re.match(r'incident:\s*(.+),\s*count:\s*(\d+)', line)
        if match:
            incident_type = match.group(1).strip().lower()
            count = int(match.group(2))
            discovered_incidents[incident_type] = count

    return discovered_incidents

def load_incident_distribution(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def create_merged_dataframe(ground_truth, discovered, total_documents):
    ground_truth_data = [
        {
            'incident_type': incident_type,
            'count': int(total_documents * percentage / 100),
            'source': 'Ground Truth'
        }
        for incident_type, percentage in ground_truth.items()
    ]
    
    discovered_data = [
        {
            'incident_type': incident_type,
            'count': count,
            'source': 'Discovered'
        }
        for incident_type, count in discovered.items()
    ]
    
    merged_df = pd.DataFrame(ground_truth_data + discovered_data)
    
    # Sort ground truth and discovered incidents separately by count
    ground_truth_sorted = merged_df[merged_df['source'] == 'Ground Truth'].sort_values('count', ascending=False)
    discovered_sorted = merged_df[merged_df['source'] == 'Discovered'].sort_values('count', ascending=False)
    
    # Concatenate the sorted dataframes
    final_df = pd.concat([ground_truth_sorted, discovered_sorted]).reset_index(drop=True)
    
    return final_df

def main():
    input_folder = 'data'
    documents = read_documents(input_folder)
    
    if not documents:
        print("No documents found in the input folder.")
        return

    # Load incident type distribution
    distribution_file = os.path.join(input_folder, "incident_type_distribution.json")
    incident_distribution = load_incident_distribution(distribution_file)

    # Analyze incidents using GPT-4
    analysis = analyze_incidents(documents)
    print("\nIncident Analysis:")
    print(analysis)

    # Parse the analysis
    discovered_incidents = parse_analysis(analysis)

    if not discovered_incidents:
        print("No valid incident types and counts were extracted from the analysis.")
        return

    # Create merged DataFrame
    total_documents = len(documents)
    merged_df = create_merged_dataframe(incident_distribution, discovered_incidents, total_documents)

    # Save merged DataFrame to CSV
    csv_file = os.path.join(input_folder, "incident_report.csv")
    merged_df.to_csv(csv_file, index=False)
    
    print(f"\nMerged incident report saved to: {csv_file}")

    # Display merged DataFrame
    print("\nMerged Incident Report:")
    print(merged_df)

if __name__ == "__main__":
    main()
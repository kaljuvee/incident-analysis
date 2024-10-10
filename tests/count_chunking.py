import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import re
from typing import List, Dict
from collections import Counter
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add logging setup at the beginning of the file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_documents(input_folder):
    all_content = ""
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(input_folder, filename), 'r') as file:
                all_content += file.read().strip() + "\n\n"
    return all_content

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text(text: str, max_tokens: int = 5000) -> List[str]:
    chunks = []
    current_chunk = ""
    current_tokens = 0

    sentences = text.split('.')
    for sentence in sentences:
        sentence_tokens = num_tokens_from_string(sentence)
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence + '.'
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def create_embedded_index(chunks: List[str]) -> List[Dict]:
    embedded_index = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embedded_index.append({"text": chunk, "embedding": embedding})
    return embedded_index

def analyze_incidents_single_turn(embedded_index: List[Dict], max_tokens: int) -> Dict[str, int]:
    prompt = """Analyze the following incident reports and identify all unique types of incidents mentioned. Then, count how many times each incident type appears. Provide the response in this exact format:
incident: <incident type>, count: <incident count>

Incident reports:
{chunk}

Provide only the list of incidents and counts, with no additional text or explanations."""

    all_incident_counts = Counter()

    for item in embedded_index:
        chunk = item["text"]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing incident reports. Respond only in the exact format specified in the prompt."},
                {"role": "user", "content": prompt.format(chunk=chunk)}
            ],
            max_tokens=max_tokens
        )
        chunk_counts = parse_analysis(response.choices[0].message.content)
        all_incident_counts.update(chunk_counts)

    return dict(all_incident_counts)

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

def main(max_tokens: int = 5000):
    input_folder = 'data'
    all_content = read_documents(input_folder)
    
    if not all_content:
        print("No content found in the input folder.")
        return

    # Load incident type distribution
    distribution_file = os.path.join(input_folder, "incident_type_distribution.json")
    incident_distribution = load_incident_distribution(distribution_file)

    # Chunk all content and create embedded index
    chunks = chunk_text(all_content, max_tokens)
    embedded_index = create_embedded_index(chunks)

    # Analyze incidents using GPT-4 with single-turn reasoning and embedded index
    discovered_incidents = analyze_incidents_single_turn(embedded_index, max_tokens)
    print("\nIncident Analysis:")
    for incident_type, count in discovered_incidents.items():
        print(f"incident: {incident_type}, count: {count}")

    if not discovered_incidents:
        print("No valid incident types and counts were extracted from the analysis.")
        return

    # Create merged DataFrame
    total_documents = len(chunks)  # Use number of chunks instead of documents
    merged_df = create_merged_dataframe(incident_distribution, discovered_incidents, total_documents)

    # Save merged DataFrame to CSV
    csv_file = os.path.join(input_folder, "incident_report_embedded_single_turn_large_chunks.csv")
    merged_df.to_csv(csv_file, index=False)
    
    print(f"\nMerged incident report saved to: {csv_file}")

    # Display merged DataFrame
    print("\nMerged Incident Report:")
    print(merged_df)

if __name__ == "__main__":
    main(max_tokens=5000)  # Set the default max_tokens to 5000
import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import logging
from tqdm import tqdm
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

# Define model variables
CHAT_MODEL = 'gpt-4'
EMBEDDING_MODEL = 'text-embedding-3-large'

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_all_documents(folder_path):
    all_documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                all_documents.append(file.read().strip())
    return all_documents

def create_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def create_aggregate_document(all_documents):
    return "\n\n".join(all_documents)

def chunk_text(text, max_tokens=1000):
    encoding = tiktoken.encoding_for_model(CHAT_MODEL)
    tokens = encoding.encode(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        if current_length + 1 > max_tokens:
            chunks.append(encoding.decode(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(token)
        current_length += 1

    if current_chunk:
        chunks.append(encoding.decode(current_chunk))

    return chunks

def analyze_incident(chunk):
    base_prompt = """Analyze the following incident report chunk and identify all unique types of incidents mentioned. Then, count how many times each incident type appears. Provide the response in this exact format:
incident: <incident type>, count: <incident count>

Incident report chunk:
{chunk}

Provide only the list of incidents and counts, with no additional text or explanations. If no incidents are found, respond with 'incident: none, count: 0'."""

    prompt = base_prompt.format(chunk=chunk)
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in analyzing incident reports. Respond only with the incident type and count."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    result = response.choices[0].message.content.strip().lower()
    
    # Ensure the result is properly formatted
    if not result or not any(line.count(', count: ') == 1 for line in result.split('\n')):
        return 'incident: none, count: 0'
    
    return result

def process_document(document):
    chunks = chunk_text(document, max_tokens=1000)
    all_results = []

    for chunk in chunks:
        result = analyze_incident(chunk)
        all_results.append(result)
    
    return merge_incident_counts(all_results)

def merge_incident_counts(results):
    merged = {}
    for result in results:
        for line in result.split('\n'):
            line = line.strip().lower()
            if not line:
                continue
            try:
                incident_part, count_part = line.split(', count: ')
                incident_type = incident_part.replace('incident: ', '').strip()
                count = int(count_part)
                merged[incident_type] = merged.get(incident_type, 0) + count
            except ValueError:
                logging.warning(f"Skipping malformed line: {line}")
    return '\n'.join([f"incident: {k}, count: {v}" for k, v in merged.items()])

def load_incident_distribution(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def create_merged_dataframe(ground_truth, discovered, total_incidents):
    ground_truth_data = [
        {
            'incident_type': incident_type,
            'count': int(total_incidents * percentage / 100),
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

def get_max_tokens():
    try:
        model_info = client.models.retrieve(CHAT_MODEL)
        return model_info.maximum_tokens
    except Exception as e:
        logging.warning(f"Couldn't retrieve max tokens for {CHAT_MODEL}. Using default value. Error: {e}")
        return 8192  # Default value for GPT-4

def batch_documents(documents, max_batch_tokens):
    encoding = tiktoken.encoding_for_model(CHAT_MODEL)
    batches = []
    current_batch = []
    current_length = 0

    for doc in documents:
        doc_tokens = len(encoding.encode(doc))
        if current_length + doc_tokens > max_batch_tokens:
            batches.append("\n\n".join(current_batch))
            current_batch = []
            current_length = 0
        current_batch.append(doc)
        current_length += doc_tokens

    if current_batch:
        batches.append("\n\n".join(current_batch))

    return batches

def process_batch(batch):
    chunks = chunk_text(batch, max_tokens=3000)  # Increased chunk size
    all_results = []

    for chunk in chunks:
        result = analyze_incident(chunk)
        all_results.append(result)
        logging.debug(f"Chunk analysis result: {result}")
    
    merged_result = merge_incident_counts(all_results)
    logging.debug(f"Merged result for batch: {merged_result}")
    return merged_result

def main():
    input_folder = 'data'
    distribution_file = os.path.join(input_folder, "incident_type_distribution.json")
    incident_distribution = load_incident_distribution(distribution_file)

    logging.info("Reading all documents...")
    all_documents = read_all_documents(input_folder)

    max_tokens = get_max_tokens()
    max_batch_tokens = max_tokens - 1000  # Reserve 1000 tokens for the prompt and response

    logging.info("Batching documents...")
    batches = batch_documents(all_documents, max_batch_tokens)

    logging.info("Analyzing incidents...")
    discovered_incidents = {}
    
    for batch in tqdm(batches, desc="Analyzing batches"):
        incident_types = process_batch(batch)
        for line in incident_types.split('\n'):
            incident_type, count = line.split(', count: ')
            incident_type = incident_type.replace('incident: ', '')
            discovered_incidents[incident_type] = discovered_incidents.get(incident_type, 0) + int(count)

    total_incidents = len(all_documents)
    logging.info(f"Total incidents analyzed: {total_incidents}")

    logging.info("Analysis complete. Creating merged DataFrame...")

    # Create merged DataFrame
    merged_df = create_merged_dataframe(incident_distribution, discovered_incidents, total_incidents)

    # Create reports directory if it doesn't exist
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save merged DataFrame to CSV in the reports folder with timestamp
    csv_file = os.path.join(reports_dir, f"incident_report_{timestamp}.csv")
    merged_df.to_csv(csv_file, index=False)
    
    logging.info(f"Merged incident report saved to: {csv_file}")

    # Display merged DataFrame
    print("\nMerged Incident Report:")
    print(merged_df)

if __name__ == "__main__":
    main()
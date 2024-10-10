import os
import json
import logging
from collections import Counter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
INDEX_DIR = "index"

def load_vector_store():
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Read the latest index timestamp
        with open(os.path.join(INDEX_DIR, "latest_index.txt"), "r") as f:
            timestamp = f.read().strip()
        
        index_name = f"index_{timestamp}"
        
        if os.path.exists(os.path.join(INDEX_DIR, f"{index_name}.faiss")) and os.path.exists(os.path.join(INDEX_DIR, f"{index_name}.pkl")):
            logging.info(f"Loading existing vector store: {index_name}")
            vector_store = FAISS.load_local(
                folder_path=INDEX_DIR, 
                index_name=index_name, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            return vector_store
        else:
            logging.error(f"Vector store not found: {index_name}")
            return None
    except Exception as e:
        logging.error(f"Error in load_vector_store: {str(e)}")
        return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def discover_and_count_incidents(retriever):
    logging.info("Discovering and counting all incident types")
    
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # Use a pre-made prompt from the hub
    prompt = hub.pull("rlm/rag-prompt")
    
    # Construct the LCEL chain
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        logging.info("Invoking qa_chain to discover incident types")
        discover_result = qa_chain.invoke("Analyze the documents and list all unique incident types mentioned. Provide the response as a comma-separated list of incident types, with no additional text or explanations.")
        incident_types = [incident_type.strip().lower() for incident_type in discover_result.split(',')]
        logging.info(f"Discovered incident types: {incident_types}")

        incident_counts = {}
        for incident_type in incident_types:
            logging.info(f"Counting occurrences of incident type: {incident_type}")
            count_result = qa_chain.invoke(f"Count how many times the incident type '{incident_type}' is mentioned in the relevant documents. Provide the response as a single number. If no incidents of the specified type are found, respond with '0'.")
            try:
                count = int(count_result.strip())
            except ValueError:
                logging.warning(f"Unable to convert result to integer for {incident_type}: {count_result}")
                count = 0
            incident_counts[incident_type] = count
            logging.info(f"Count for '{incident_type}': {count}")

        return incident_counts
    except Exception as e:
        logging.error(f"Error in discover_and_count_incidents: {str(e)}")
        return {}

def main():
    logging.info("Starting incident analysis process")
    
    # Load vector store
    vector_store = load_vector_store()
    if not vector_store:
        logging.error("Failed to load vector store. Cannot proceed with the analysis.")
        return

    # Discover and count incidents
    discovered_incidents = discover_and_count_incidents(vector_store.as_retriever())

    # Print total counts
    logging.info("Printing total incident counts")
    print("Total incident counts:")
    for incident_type, count in discovered_incidents.items():
        print(f"{incident_type}: {count}")
    
    # Save results to a JSON file
    output_file = "data/incidents_discovered.json"
    with open(output_file, 'w') as f:
        json.dump(discovered_incidents, f, indent=2)
    logging.info(f"Results saved to {output_file}")

    logging.info("Incident analysis process completed")

if __name__ == "__main__":
    main()
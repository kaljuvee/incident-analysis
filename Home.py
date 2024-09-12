import streamlit as st

st.markdown('''

    # RAG Modeling Steps for Incident Analysis

    This document outlines the sequence of steps involved in the Retrieval-Augmented Generation (RAG) modeling process for incident analysis. Each step is crucial for building an effective system that can analyze incident reports and answer questions based on the data.

    ## 1. Document Loading

    **Purpose**: Loads incident reports from JSON files in a specified directory.

    **Why it's useful**: 
    - Centralizes data collection from multiple incident reports
    - Ensures all relevant information is available for analysis
    - Provides a structured format for consistent processing

    ## 2. Document Processing (Optional Chunking)

    **Purpose**: Breaks down large documents into smaller, manageable chunks.

    **Why it's useful**:
    - Improves retrieval accuracy by allowing more granular matching
    - Enables handling of long documents that might exceed model token limits
    - Facilitates more precise context selection for question answering

    ## 3. Embedding Creation

    **Purpose**: Generates vector representations (embeddings) of the documents or chunks using a selected language model.

    **Why it's useful**:
    - Converts text into a format that machines can easily process and compare
    - Captures semantic meaning, allowing for more accurate similarity searches
    - Enables efficient retrieval of relevant information

    ## 4. FAISS Index Creation

    **Purpose**: Builds a FAISS index from the document embeddings for efficient similarity search.

    **Why it's useful**:
    - Dramatically speeds up similarity searches in large document collections
    - Enables real-time retrieval of relevant documents for user queries
    - Scales well to handle growing datasets

    ## 5. Incident Type Extraction

    **Purpose**: Analyzes documents to extract relevant incident types.

    **Why it's useful**:
    - Automatically categorizes incidents without manual labeling
    - Identifies common themes and patterns across incident reports
    - Provides a structured way to analyze and compare different types of incidents

    ## 6. Incident Counting

    **Purpose**: Counts the occurrences of each incident type across all documents.

    **Why it's useful**:
    - Quantifies the frequency of different incident types
    - Helps identify the most common safety issues
    - Provides data for prioritizing safety measures and resource allocation

    ## 7. Pattern Analysis

    **Purpose**: Performs various analyses on the incident data.

    **Why it's useful**:
    - Reveals trends and patterns in incident occurrences over time
    - Identifies potential correlations between different types of incidents
    - Provides insights for predictive modeling and preventive measures
    - Helps in understanding the safety performance of different plants or locations

    ## 8. Question Answering with RAG

    **Purpose**: Retrieves relevant document chunks based on a user's question and generates an answer based on the retrieved context.

    **Why it's useful**:
    - Provides quick, context-aware answers to user queries
    - Leverages the entire knowledge base to give comprehensive responses
    - Improves the accuracy of answers by using relevant context
    - Enables interactive exploration of the incident data
            
# GraphRAG CLI
## Set Workspace Variables

```
    python -m graphrag.index --init --root ./data
```

## Running the Indexing pipeline

```
    python -m graphrag.index --root ./data
```

## Running the Query Engine

```
    python -m graphrag.query \
    --root ./data \
    --method global \
    "How many heat related incidents were there?"
```

```
    python -m graphrag.query \
    --root ./data \
    --method local \
    "What are the most common types of incidents?"
```

''')
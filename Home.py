import streamlit as st

st.markdown('''

# Incident Analysis
            
This sample RAG interface provides a basic implementation to address the customer's use cases. Here's a breakdown of the main components:

## Main Components
            
- **Document Retrieval:** We use TF-IDF vectorization and cosine similarity to find relevant documents based on a query.
- **Question Answering:** We use a pre-trained question-answering model to extract specific information from the relevant documents.
- **Incident Type Counting:** We implement a simple keyword-based counting method, which can be easily extended to include new incident types.
- **Pattern Analysis:** We provide basic functionality to analyze incident rates per plant, changes over time, and correlations between incident types.

## Modelling Steps
            
- **Speed:** TF-IDF offers the fastest retrieval times.
- **Generalization:**
1. Embedding models, especially BERT and its variants, offer better generalization to new incident types and descriptions.
2. The keyword-based counting can be easily extended to include new incident types without major changes to the code.
- **Pattern Recognition:** Functionality to provide a starting point for identifying trends and correlations in the data.

## Areas for Improvement
            
- The incident type counting is still keyword-based and might miss some nuances in the descriptions.
- This implementation doesn't address the scalability issues that might arise with millions of documents.

## Further Improvements
            
To further improve this system, you might consider:
            
- Using more advanced NLP techniques for incident classification, such as fine-tuning a language model on your specific domain.
- Implementing a database solution for efficient storage and retrieval of documents.
- Developing a more comprehensive analytics pipeline for pattern recognition, possibly incorporating machine learning models for prediction and anomaly detection.
- Setting up a system for continuous learning and updating of the model as new incident types and patterns emerge.
            
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
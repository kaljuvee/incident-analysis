# Overview

### Clone the Repo
  ```
  git clone https://github.com/kaljuvee/incident-analysis.git
  ```

### Create and Activate the Virtual Environment

- Set up a Python virtual environment and activate it (Windows/VS Code / Bash new Terminal):
  ```
  python -m venv venv
  source venv/Scripts/activate
  ```
  - Set up a Python virtual environment and activate it (Linux):
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```
  
- Install dependencies from the `requirements.txt` file:
  ```
  pip install -r requirements.txt
  ```

  ### Run the app
  - In VS Code Bash terminal run:
  ```
  streamlit run Home.py
  ```

  ## Running Indexer


```
    mkdir -p ./data/incidents/
``` 
- add documents to this folder

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
    --root ./ragtest \
    --method global \
    "How many heat related incidents were there?"
```

```
    python -m graphrag.query \
    --root ./ragtest \
    --method local \
    "What are the most common types of incidents?"
```
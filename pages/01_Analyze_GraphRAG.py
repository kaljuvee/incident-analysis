import streamlit as st
import subprocess
import os

st.set_page_config(page_title="GraphRAG Query Engine", layout="wide")

st.title("GraphRAG Query Engine")

col1, col2 = st.columns([1, 1])

with col1:
    root_path = st.text_input("Root Path", value="./data/incidents")
    query = st.text_area("Enter your query:", height=100)
    method = st.radio("Select query method:", ("global", "local"))

    if st.button("Run Query"):
        if not query:
            st.warning("Please enter a query.")
        elif not os.path.exists(root_path):
            st.error(f"Workspace not found at {root_path}. Please check the path.")
        else:
            with st.spinner("Processing query..."):
                try:
                    # Construct the command
                    command = [
                        "python", "-m", "graphrag.query",
                        "--root", root_path,
                        "--method", method,
                        query
                    ]
                    
                    # Run the command and capture output
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    
                    st.subheader("Query Result:")
                    st.text_area("Result", value=result.stdout, height=300)
                    
                    if result.stderr:
                        st.warning("Warnings or additional information:")
                        st.text(result.stderr)
                except subprocess.CalledProcessError as e:
                    st.error(f"An error occurred while running the query: {e}")
                    if e.stdout:
                        st.text("Standard output:")
                        st.text(e.stdout)
                    if e.stderr:
                        st.text("Standard error:")
                        st.text(e.stderr)

with col2:
    st.subheader("Instructions")
    st.markdown("""
    1. Enter the root path where your indexed documents are located.
    2. Type or paste your query into the text area.
    3. Select the query method (global or local).
    4. Click "Run Query" to see the results.
    """)

    st.subheader("Example Queries")
    st.markdown("""
    **Global method example:**
    ```
    How many heat related accidents were there?
    ```
    
    **Local method example:**
    ```
    "What are the most common types of incidents?"
    ```
    """)
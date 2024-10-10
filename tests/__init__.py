import tiktoken

def get_encoding():
    return tiktoken.get_encoding("cl100k_base")

def chunk_text(text, max_tokens=5000):
    encoding = get_encoding()
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

# You may need to update other functions that use the chunking functionality
# For example, if there's a function like process_text, update it to use the new chunk_text function:

def process_text(text):
    chunks = chunk_text(text)
    # Process each chunk
    for chunk in chunks:
        # Your processing logic here
        pass

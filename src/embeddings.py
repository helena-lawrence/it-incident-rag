
from sentence_transformers import SentenceTransformer

from document_loader import load_documents
from chunking import chunk_documents


# Step 1: Load the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


# Step 2: Create embedding function
def generate_embeddings(chunks):

    # Extract text from each chunk
    texts = [chunk["content"] for chunk in chunks]

    # Convert text into embeddings
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


# Step 3: Run the program
if __name__ == "__main__":

    # Load documents
    documents = load_documents()

    # Split documents into chunks
    chunks = chunk_documents(documents)

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    print("\nTotal chunks:", len(chunks))

    print("Embedding shape:", embeddings.shape)

    print("\nFirst chunk:")
    print(chunks[0]["content"])

    print("\nFirst embedding (first 10 values):")
    print(embeddings[0][:10])

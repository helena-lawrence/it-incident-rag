
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_documents


def chunk_documents(documents):

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    # Loop through each document
    for document in documents:

        # Extract text from document
        text = document["content"]

        # Split text into smaller chunks
        chunks = text_splitter.split_text(text)

        # Store each chunk with metadata
        for index, chunk in enumerate(chunks):

            chunk_data = {
                "content": chunk,
                "source": document["source"],
                "file_type": document["file_type"],
                "chunk_id": index
            }

            all_chunks.append(chunk_data)

    return all_chunks


# Run the program
if __name__ == "__main__":

    documents = load_documents()

    chunks = chunk_documents(documents)

    print(f"\nTotal chunks created: {len(chunks)}")

    for chunk in chunks:

        print("\n-----------------------")
        print("Source:", chunk["source"])
        print("Chunk ID:", chunk["chunk_id"])
        print("Content:")
        print(chunk["content"])

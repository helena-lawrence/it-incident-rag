
from pathlib import Path
from pypdf import PdfReader


# Step 1: Find project and data folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# Step 2: Create document loading function
def load_documents():

    documents = []

    # Step 3: Read each file
    for file_path in sorted(DATA_DIR.iterdir()):

        if not file_path.is_file():
            continue

        print(f"Processing: {file_path.name}")

        # Step 4: Read TXT and Markdown files
        if file_path.suffix.lower() in [".txt", ".md"]:

            text = file_path.read_text(encoding="utf-8")

        # Step 5: Read PDF files
        elif file_path.suffix.lower() == ".pdf":

            try:
                reader = PdfReader(file_path)

                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )

            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")
                continue

        # Step 6: Skip unsupported file types
        else:
            continue

        # Step 7: Skip empty documents
        if not text.strip():
            print(f"Skipping empty document: {file_path.name}")
            continue

        # Step 8: Store content and metadata
        document = {
            "content": text,
            "source": file_path.name,
            "file_type": file_path.suffix.lower()
        }

        documents.append(document)

    return documents


# Step 9: Run the loader
if __name__ == "__main__":

    documents = load_documents()

    print(f"\nTotal documents loaded: {len(documents)}")

    for document in documents:

        print("\n-----------------------")
        print("Source:", document["source"])
        print("Type:", document["file_type"])
        print("Content:")
        print(document["content"][:300])

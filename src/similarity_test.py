
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "My company VPN is not connecting.",
    "I cannot establish a VPN connection.",
    "How can I reset my password?",
    "Outlook is not receiving emails."
]

embeddings = model.encode(
    sentences,
    convert_to_tensor=True
)

query_embedding = embeddings[0]

for index in range(1, len(sentences)):

    similarity = util.cos_sim(
        query_embedding,
        embeddings[index]
    ).item()

    print(
        f"Similarity with '{sentences[index]}': "
        f"{similarity:.4f}"
    )

from pinecone import Pinecone, ServerlessSpec
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "speech-therapy"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # placeholder, will match embedding size later
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


def store_reference(word, phonemes, embedding):
    index.upsert([
        {
            "id": f"{word}_ref",
            "values": embedding,
            "metadata": {
                "word": word,
                "phonemes": phonemes
            }
        }
    ])


def compare_pronunciation(embedding):
    result = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True
    )
    return result

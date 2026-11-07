from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

document = Path("data/company_faq.txt").read_text(
    encoding="utf-8"
)


chunks = [chunk.strip()
          for chunk in document.split("\n\n")
          if chunk.strip()]


# Create metadata for every document chunk
metadata = [
    {
        "source": "company_faq.txt",
        "chunk_id": i + 1
    }
    for i in range(len(chunks))
]



from sentence_transformers import SentenceTransformer
# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)

embeddings_path = cache_dir / "embeddings.npy"
chunks_path = cache_dir / "chunks.json"

# Check if our saved files exist
if embeddings_path.exists() and chunks_path.exists():

    with open(chunks_path, "r", encoding="utf-8") as f:
        cached_chunks = json.load(f)

    if cached_chunks == chunks:
        embeddings = np.load(embeddings_path)
        print("Loaded cached embeddings!")

    else:
        embeddings = model.encode(chunks)
        np.save(embeddings_path, embeddings)

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print("Documents changed. Cache updated!")

else:
    embeddings = model.encode(chunks)
    np.save(embeddings_path, embeddings)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Created embeddings cache!")


def retrieve(query, top_k=2,threshold = 0.35):

    # Convert the user's question into an embedding
    query_embedding = model.encode(query)

    # Calculate cosine similarity against all chunks
    similarities = (
        embeddings @ query_embedding
        / (
            np.linalg.norm(embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )
    )

    # Find indices of the most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return the matching chunks and scores
    results = [
        (chunks[i], float(similarities[i]), metadata[i])
        for i in top_indices
        if similarities[i] >= threshold
    ]


    return results

# def inspect_retrieval(questions):

#     for question in questions:
#         results = retrieve(question, top_k=2, threshold=0.35)

#         print(f"\nQuestion: {question}")

#         if not results:
#             print("No relevant documents retrieved.")
#             continue

#         for chunk, score, meta in results:
#             print(
#                 f"Chunk {meta['chunk_id']} | "
#                 f"Similarity: {score:.3f}"
#             )
#             print(f"Text: {chunk[:100]}...")


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


def generate_answer(question):

   
    # Step 1: Retrieve relevant chunks once
    results = retrieve(question, top_k=2)
    if not results:
        return "I don't know based on the provided documents.", []

    # Step 2: Build context with source identifiers
    context = "\n\n".join(
        f"[Source {i+1}] {chunk}"
        for i, (chunk, score, meta) in enumerate(results)
    )

    # Step 3: Generate answer
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a company knowledge assistant. "
                    "Answer questions using ONLY the provided context. "
                    "If the context does not contain the answer, "
                    "say 'I don't know based on the provided documents'. "
                    "Do not invent information. "
                    "Cite supporting context using [Source 1], "
                    "[Source 2], etc."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    # Return both the answer and the sources
    return answer, results




if __name__ == "__main__":

    print("Welcome to NovaTech RAG Assistant!")

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer, sources = generate_answer(question)

        print("\nAnswer:", answer)

        print("\nRetrieved Sources:")
        for i, (chunk, score,meta) in enumerate(sources, start = 1):
             print(
                f"[Source {i}] "
                f"- {meta['source']} | "
                f"Chunk {meta['chunk_id']} | "
                f"Similarity: {score:.3f}"
            )

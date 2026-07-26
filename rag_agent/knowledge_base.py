import os

import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "fraud_kb")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "fraud_regulations"

CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def build_knowledge_base(kb_dir=KB_DIR, persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME, reset=True):
    client = chromadb.PersistentClient(path=persist_directory)

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)

    documents, metadatas, ids = [], [], []
    for filename in sorted(os.listdir(kb_dir)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(kb_dir, filename), encoding="utf-8") as f:
            text = f.read()

        for i, chunk in enumerate(chunk_text(text)):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})
            ids.append(f"{filename}::chunk_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def query_knowledge_base(query, n_results=3, persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(collection_name)

    results = collection.query(query_texts=[query], n_results=n_results)

    matches = []
    for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        matches.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": distance,
        })
    return matches


if __name__ == "__main__":
    collection = build_knowledge_base()
    print(f"Knowledge base built: {collection.count()} chunks in collection '{COLLECTION_NAME}'")

    test_query = "ACH unauthorized debit regulations"
    print(f"\nTest query: {test_query!r}\n")

    for rank, match in enumerate(query_knowledge_base(test_query, n_results=3), start=1):
        print(f"--- Result {rank} (source: {match['source']}, chunk {match['chunk_index']}, distance: {match['distance']:.4f}) ---")
        print(match["text"][:400].strip() + "...")
        print()

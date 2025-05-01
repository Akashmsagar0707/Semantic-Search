# Please note :-

# Please place the PDF files in the same folder as the script.
# The script automatically detects and reads all .pdf files from the current directory without needing to specify paths.

import argparse
import os
from pathlib import Path

import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


def extract_text(path: str) -> str:

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_embeddings(chunks: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings, model


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def semantic_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: list[str],
    sources: list[str],
    top_k: int = 6
) -> list[dict]:

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(score),
            "source": sources[idx],
            "text": chunks[idx].replace("\n", " ").strip()
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="In-memory semantic search over PDF documents using FAISS"
    )
    parser.add_argument(
        "pdf_paths", nargs="*",
        help="Paths or glob patterns to PDF files............."
    )
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Max chars per text chunk (default:1000)")
    parser.add_argument("--overlap",    type=int, default=200,
                        help="Overlap chars between chunks (default:200)")
    parser.add_argument("--top_k",      type=int, default=5,
                        help="Number of top results per query (default:5)")
    args = parser.parse_args()

    # Auto-detect PDFs if none specified
    if not args.pdf_paths:
        pdf_files = [str(p) for p in Path(__file__).parent.glob("*.pdf")]
        if not pdf_files:
            print("no pdf found.")
            return
    else:
        pdf_files = args.pdf_paths


    all_chunks = []
    all_sources = []


    for pdf in pdf_files:
        text = extract_text(pdf)


        print(f"\n\n=== Preview of '{pdf}' ({len(text)} chars) ===")
        print(text[:8000].replace("\n", " "))
        print("…\n")
        #


        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)

        # Accumulate both chunks and their source filenames------------------------------------

        all_chunks.extend(chunks)
        all_sources.extend([os.path.basename(pdf)] * len(chunks))
        word_count = len(text.split())
        char_count = len(text)
        print(f"Total characters: {char_count}")

        print(f"'{pdf}' contains {word_count} words\n")
        line_count = text.count('\n') + 1
        print(f"Total lines: {line_count}")

        # ------------Semantic‐search ------------------------------------------------
    print("\n🧠 Setting up semantic search.........")
    embeddings, model = build_embeddings(all_chunks)
    index = build_faiss_index(embeddings)
    print(f"✅ Ready: indexed {len(all_chunks)} chunks.")


    while True:
        query = input("Query> ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break
        results = semantic_search(query, model, index, all_chunks, all_sources, top_k=args.top_k)
        print(f"\nResults for “{query}”:")
        for i, hit in enumerate(results, 1):
            print(f"[{i}] {hit['source']} (score {hit['score']:.3f})")
            print(hit['text'], "\n")

if __name__ == "__main__":
    main()
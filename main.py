from flask import Flask, render_template, request, jsonify
import os
from pathlib import Path
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Global variables to store the search index and model
model = None
index = None
chunks = []
sources = []


def initialize_system():
    global model, index, chunks, sources
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.IndexFlatIP(384)  # Default dimension for MiniLM-L6
    chunks = []
    sources = []


def process_pdf(pdf_path):
    global index, chunks, sources

    # Extract text
    text = extract_text(pdf_path)
    file_chunks = chunk_text(text)

    # Create embeddings
    embeddings = model.encode(
        file_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Update index
    if index.ntotal == 0:
        index.add(embeddings)
    else:
        index.add(embeddings)

    # Update metadata
    chunks.extend(file_chunks)
    sources.extend([os.path.basename(pdf_path)] * len(file_chunks))


def extract_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def setup():
    initialize_system()
    pdf_files = [str(p) for p in Path(__file__).parent.glob("*.pdf")]
    for pdf in pdf_files:
        process_pdf(pdf)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and file.filename.endswith('.pdf'):
        filename = os.path.join(Path(__file__).parent, file.filename)
        file.save(filename)
        process_pdf(filename)
        return jsonify({'message': f'File {file.filename} uploaded and processed successfully'})

    return jsonify({'error': 'Invalid file type'})


@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    top_k = request.json.get('top_k', 5)

    if not query or index is None:
        return jsonify([])

    # Encode query
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Search
    distances, indices = index.search(q_emb, top_k)

    # Prepare results
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx >= 0 and idx < len(chunks):  # Handle possible invalid indices
            results.append({
                "score": float(score),
                "source": sources[idx],
                "text": chunks[idx].replace("\n", " ").strip()
            })

    return jsonify(results)


if __name__ == '__main__':
    with app.app_context():
        setup()
    app.run(debug=True)
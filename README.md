# 🔍 Semantic PDF Search with FAISS and Sentence Transformers

This Python script allows you to perform **semantic search** on the contents of `.pdf` files using Sentence Transformers and FAISS. It extracts text from PDF files, splits the text into overlapping chunks to preserve context, generates embeddings, and performs fast similarity search for user queries.

---

## 📁 Project Structure

- `main.py`: The core script to run the semantic search.
- `.pdf` files: Place your PDF documents in the same folder as `main.py`.

> **Note:**  
> The script will automatically detect and read all `.pdf` files from the current directory — no need to specify paths manually.

---

## 🧠 Features

- Extracts text from multiple PDF files.
- Splits text into overlapping chunks to preserve context.
- Converts text chunks into semantic embeddings.
- Uses FAISS for fast vector similarity search.
- Interactive command-line interface for querying.

---

## ✅ Requirements

Install the required dependencies using pip:

```bash
pip install -r requirements.txt

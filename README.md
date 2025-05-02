# sdkGPT

A powerful AI-powered assistant that lets you ask questions about your SDK or codebase using GPT-4. Supports terminal and GUI modes, fuzzy search, and automatic SDK indexing.

---

## Features

- Recursive loading of `.txt`, `.md`, `.py`, `.cpp`, `.h`, `.html`, and `.json` files
- GPT-4 powered Q&A over your local SDK
- Command-line interface with colorized output
- PyQt6-based GUI application
- Fuzzy search support for improved question matching
- Persistent vector index using FAISS
- Docker and Docker Compose support

---

## Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Add Your SDK Files

Place all relevant files into the `sdk_docs/` folder.

### 3. Build Vector Index

Before querying, run:

```bash
python prep_vectorstore.py
```

This creates `vectorstore.pkl` used by both CLI and GUI.

---

## Usage

### CLI Mode

```bash
python main.py
```

Use natural language to ask questions like:

- "How do I initialize FooSDK?"
- "What does foo_create() return?"

Type `exit` to quit.

---

### GUI Mode

```bash
python gui.py
```

A user-friendly interface will open. Type questions and get answers instantly.

---

## Docker

### Build and Run with Docker

```bash
docker build -t sdk-gpt .
docker run -it --rm -v $(pwd)/sdk_docs:/app/sdk_docs sdk-gpt
```

### Or Use Docker Compose

```bash
docker-compose up
```

---

## License

MIT License

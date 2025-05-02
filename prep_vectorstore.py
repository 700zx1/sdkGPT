from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

def custom_loader(path):
    supported_exts = ['.txt', '.md', '.py', '.cpp', '.h', '.c', '.html', '.json']
    documents = []
    for filepath in Path(path).rglob("*"):
        if filepath.suffix.lower() in supported_exts:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(filepath, "r", encoding="latin-1") as f:
                        text = f.read()
                except Exception as e:
                    print(f"\033[91m[Skipped] {filepath} (encoding error: {e})\033[0m")
                    continue
            documents.append(Document(page_content=text, metadata={"source": str(filepath)}))
            print(f"\033[94mLoaded:\033[0m {filepath}")
    return documents

def main():
    docs_path = "./sdk_docs"
    if not os.path.exists(docs_path):
        print("Directory sdk_docs/ does not exist.")
        return

    print("Loading and embedding documents...")
    documents = custom_loader(docs_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save the vectorstore using FAISS's save_local method
    vectorstore.save_local("vectorstore")

    print("Vectorstore saved to the 'vectorstore' directory")

    # Load the vectorstore using FAISS's load_local method with the safety flag
    #vectorstore = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

if __name__ == "__main__":
    main()

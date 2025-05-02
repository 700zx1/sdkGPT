from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
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
    print("\033[1m\033[96m[sdkGPT]\033[0m Starting...")

    docs_path = "./sdk_docs"
    if not os.path.exists(docs_path):
        print("\033[91mError:\033[0m Directory sdk_docs/ does not exist.")
        return

    print("\n\033[92m[Step 1]\033[0m Reading SDK files...")
    documents = custom_loader(docs_path)

    print(f"\033[92m[Step 2]\033[0m Loaded {len(documents)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    print("\033[92m[Step 3]\033[0m Creating vector index...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    print("\n\033[96mReady! Type your questions about the SDK. (type 'exit' to quit)\033[0m")
    while True:
        query = input("\n\033[1m>\033[0m ")
        if query.strip().lower() in ("exit", "quit"):
            print("\n\033[93mExiting sdkGPT. Goodbye!\033[0m")
            break
        result = qa(query)
        print("\n\033[92mAnswer:\033[0m", result['result'])

if __name__ == "__main__":
    main()

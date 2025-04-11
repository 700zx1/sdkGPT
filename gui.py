import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt6.QtCore import Qt
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from pathlib import Path
import os
import pickle

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
                    print(f"[Skipped] {filepath} (encoding error: {e})")
                    continue
            documents.append(Document(page_content=text, metadata={"source": str(filepath)}))
    return documents

class SDKGPTApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDK-GPT QA")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask about the SDK...")
        layout.addWidget(self.question_input)

        self.ask_button = QPushButton("Ask")
        layout.addWidget(self.ask_button)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

        self.setLayout(layout)
        self.ask_button.clicked.connect(self.ask_question)

        self.qa = self.load_qa()

    def load_qa(self):
        if not os.path.exists("vectorstore.pkl"):
            self.output_area.append("[!] vectorstore.pkl not found. Please run prep_vectorstore.py first.")
            return None
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
        return RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4"),
            retriever=vectorstore.as_retriever()
        )

    def ask_question(self):
        query = self.question_input.text()
        if not query.strip() or self.qa is None:
            return
        result = self.qa(query)
        self.output_area.append(f"<b>Q:</b> {query}")
        self.output_area.append(f"<b>A:</b> {result['result']}\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SDKGPTApp()
    window.show()
    sys.exit(app.exec())

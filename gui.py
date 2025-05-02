import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
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
        self.setWindowTitle("sdkGPT")
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

        # Add a new context window for displaying processed files
        self.context_window = QTextEdit()
        self.context_window.setReadOnly(True)
        self.context_window.setPlaceholderText("Processed files will be listed here...")
        layout.addWidget(self.context_window)

        # Add a loading indicator
        self.loading_label = QLabel("Loading...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("color: blue; font-weight: bold;")
        self.loading_label.hide()  # Initially hidden
        layout.addWidget(self.loading_label)

        self.setLayout(layout)
        self.ask_button.clicked.connect(self.ask_question)

        self.qa = self.load_qa()

    def load_qa(self):
        # Check if the vectorstore directory exists
        if not os.path.exists("vectorstore"):
            self.output_area.append("[!] Vectorstore directory not found. Please run prep_vectorstore.py first.")
            return None

        try:
            # Load the vectorstore using FAISS.load_local
            vectorstore = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

            # Extract and display the list of processed files
            self.display_processed_files(vectorstore)

            return RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4"),
                retriever=vectorstore.as_retriever()
            )
        except Exception as e:
            self.output_area.append(f"[!] Error loading vectorstore: {e}")
            return None

    def display_processed_files(self, vectorstore):
        """Display the list of processed files in the context window."""
        try:
            # Access all documents directly from the vectorstore's docstore
            documents = vectorstore.docstore._dict.values()  # Access all stored documents
            file_list = [doc.metadata.get("source", "Unknown") for doc in documents]

            # Display the file list in the context window
            self.context_window.clear()
            self.context_window.append("Processed Files:\n")
            for file in file_list:
                self.context_window.append(f"- {file}")
        except Exception as e:
            self.context_window.append(f"[!] Error displaying processed files: {e}")

    def ask_question(self):
        query = self.question_input.text()
        if not query.strip() or self.qa is None:
            return

        # Show the loading indicator
        self.loading_label.show()
        self.output_area.append(f"<b>Q:</b> {query}")

        try:
            # Process the query
            result = self.qa.run(query)
            self.output_area.append(f"<b>A:</b> {result}\n")
        except Exception as e:
            self.output_area.append(f"[!] Error processing query: {e}")
        finally:
            # Hide the loading indicator
            self.loading_label.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SDKGPTApp()
    window.show()
    sys.exit(app.exec())

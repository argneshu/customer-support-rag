#!/usr/bin/env python3
"""
Ask a question using knowledge loaded from a PDF file.
"""
import os
from dotenv import load_dotenv
from src.rag_system import CustomerSupportRAG
import sys

# Step 1: Load environment variables (API key)
load_dotenv()

def main():
    # Step 2: Get API key
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("❌ Error: Please set API_KEY in your .env file")
        return

    # Step 3: Get PDF file path from command line or prompt
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = input("Enter path to PDF file: ")

    # Step 4: Initialize RAG system
    rag_system = CustomerSupportRAG(api_key=api_key)

    # Step 5: Load knowledge base from PDF
    rag_system.load_knowledge_base_from_pdf(pdf_file)

    # Step 6: Get user question
    if len(sys.argv) > 2:
        question = " ".join(sys.argv[2:])
    else:
        question = input("Enter your question: ")

    # Step 7: Query the knowledge base
    result = rag_system.generate_response(question)

    # Step 8: Print the answer and details
    print(f"\n🤖 AI Response: {result['response']}")
    print(f"📈 Confidence: {result['confidence']}")
    print(f"🎯 Sources: {len(result['sources'])}")
    print(f"⚙️  Model: {result['model_used']}")
    if 'total_tokens' in result:
        print(f"🔢 Tokens used: {result['total_tokens']}")

if __name__ == "__main__":
    main()

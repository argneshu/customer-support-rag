#!/usr/bin/env python3
"""
Ask a single question to the RAG system from the command line.
"""
import os
from dotenv import load_dotenv
from src.rag_system import CustomerSupportRAG
import sys

# Load environment variables
load_dotenv()

def main():
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("❌ Error: Please set API_KEY in your .env file")
        return

    # Accept question from command line argument or prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    rag_system = CustomerSupportRAG(api_key=api_key)
    # Load the knowledge base
    rag_system.load_knowledge_base_from_csv("data/test_kb.csv")
    # Get the answer
    result = rag_system.generate_response(question)

    print(f"\n🤖 AI Response: {result['response']}")
    print(f"📈 Confidence: {result['confidence']}")
    print(f"🎯 Sources: {len(result['sources'])}")
    print(f"⚙️  Model: {result['model_used']}")
    if 'total_tokens' in result:
        print(f"🔢 Tokens used: {result['total_tokens']}")

if __name__ == "__main__":
    main()

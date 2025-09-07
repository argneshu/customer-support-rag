#!/usr/bin/env python3
"""
Simple test script for the RAG system
"""
import os
from dotenv import load_dotenv
from src.rag_system import CustomerSupportRAG
import pandas as pd

# Load environment variables
load_dotenv()

def create_sample_data():
    """Create a simple CSV file with test data"""
    sample_data = [
        {
            "question": "How do I reset my password?",
            "answer": "To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and you'll receive a reset link within 5 minutes.",
            "category": "Account Management",
            "tags": "password,login,account"
        },
        {
            "question": "What are your business hours?",
            "answer": "Our customer support is available Monday through Friday, 9 AM to 6 PM EST. We also offer 24/7 chat support for urgent issues.",
            "category": "General Information",
            "tags": "hours,support,contact"
        },
        {
            "question": "How do I cancel my subscription?",
            "answer": "You can cancel your subscription anytime from your account settings. Go to 'Billing' > 'Manage Subscription' > 'Cancel'. Your access will continue until the end of your current billing period.",
            "category": "Billing",
            "tags": "subscription,cancel,billing"
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv("data/test_kb.csv", index=False)
    print("✅ Created test_kb.csv with sample data")
    return "data/test_kb.csv"

def main():
    # Get API key from environment
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("❌ Error: Please set API_KEY in your .env file")
        return
    
    print("🚀 Starting RAG System Test...")
    
    # Initialize RAG system
    rag_system = CustomerSupportRAG(api_key=api_key)
    
    # Create and load test data
    csv_file = create_sample_data()
    rag_system.load_knowledge_base_from_csv(csv_file)
    
    # Test queries
    test_questions = [
        "I forgot my password, can you help?",
        "What time do you close?",
        "I want to cancel my account"
    ]
    
    print("\n" + "="*50)
    print("🧪 TESTING RAG SYSTEM")
    print("="*50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}] Question: {question}")
        print("-" * 40)
        
        result = rag_system.generate_response(question)
        
        print(f"🤖 AI Response: {result['response']}")
        print(f"📈 Confidence: {result['confidence']}")
        print(f"🎯 Sources: {len(result['sources'])}")
        print(f"⚙️  Model: {result['model_used']}")
        
        if 'total_tokens' in result:
            print(f"🔢 Tokens used: {result['total_tokens']}")

if __name__ == "__main__":
    main()
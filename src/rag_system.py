import os
import chromadb
import anthropic
import pandas as pd
from typing import List, Dict
import uuid
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class CustomerSupportRAG:
    def __init__(self, api_key: str, persist_directory: str = "./chroma_db"):  #Creates our main RAG system class. When you create an instance, you pass in your API key and where to store the database.
        self.llm_client = anthropic.Anthropic(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory) #Creates a local vector database in the ./chroma_db folder
        self.collection = self.chroma_client.get_or_create_collection( #Makes a "collection" (like a table) called "customer_support_kb"
            name="customer_support_kb",
            #This is where all your customer support Q&As will be stored as vectors
            metadata={"description": "Customer support knowledge base"}
        )
        self.llm_model = "claude-sonnet-4-20250514"  # 
        print(f"✅ RAG system initialized with model: {self.llm_model}")
    
    #What it does:Takes text (like "How do I reset my password?")
    #Converts it into numbers (vectors) that capture the meaning
    #Uses a free, local model so no extra API costs
    #Returns lists of numbers that represent the semantic meaning

    #How This All Works Together:

    #You ask a question → "How do I cancel my subscription?"
    #Embedding converts it to numbers → [0.1, -0.3, 0.8, ...]
    #Database finds similar stored Q&As → Finds "How to cancel subscription"
    #Claude Sonnet 4 reads the context → Gets the stored answer
    #Claude generates a response → Gives you a helpful, accurate answer

    def create_embeddings_free(self, texts: List[str]) -> List[List[float]]:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts, convert_to_tensor=False, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ Error creating embeddings: {str(e)}")
            # Fallback: create dummy embeddings for testing
            print("⚠️  Using dummy embeddings for testing...")
            import numpy as np
            return [np.random.rand(384).tolist() for _ in texts]
    


    
    def load_knowledge_base_from_csv(self, csv_file_path: str):
        """
        Load customer support knowledge base from a CSV file
        Expected columns: 'question', 'answer', 'category', 'tags'
        """
        try:
            df = pd.read_csv(csv_file_path) #Reads the CSV file into a pandas DataFrame (like a spreadsheet)
            required_columns = ['question', 'answer', 'category'] #Checks if the CSV has the required columns: question, answer, category
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}") #If missing columns, stops and shows an error
            
            # Prepare documents for embedding ,  Prepare Data Containers
            documents = [] # Will hold the text to be embedded
            metadatas = [] # Will hold extra info about each document
            ids = [] # Will hold unique IDs for each document
            
            for index, row in df.iterrows(): #Goes through each row in the CSV file
                # Combine question and answer for better context
                document = f"Question: {row['question']}\nAnswer: {row['answer']}" #Combines the question and answer into one text block
                documents.append(document)
                
                # Prepare metadata
                metadata = {
                    "question": row['question'], #Stores extra information about each Q&A pair
                    "answer": row['answer'],
                    "category": row['category'],
                    "tags": row.get('tags', ''), #row.get('tags', '') means "get tags if they exist, otherwise use empty string"
                    "source": "knowledge_base"
                }
                metadatas.append(metadata)
                ids.append(f"kb_{index}") #Creates unique IDs like "kb_0", "kb_1", "kb_2" for each document
            
            # Create embeddings using local model
            print("Creating embeddings...") #Takes all the combined Q&A texts
            
            # Converts them into vectors (lists of numbers) that capture meaning, Example: "How do I reset password?" becomes [0.1, -0.3, 0.8, ...]
            embeddings = self.create_embeddings_free(documents) 
            # Add to Chroma collection,  Store in Database
            self.collection.add(
                documents=documents,    #Saves everything to the Chroma vector database, Now you can search through this data using semantic similarity
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Successfully loaded {len(documents)} documents into knowledge base")
            
        except Exception as e:
            print(f"❌ Error loading knowledge base: {str(e)}")


    def search_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Search the knowledge base for relevant documents
        
        Args:
            query: User's question
            n_results: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        # Create embedding for the user's query
        query_embedding = self.create_embeddings_free([query])[0] #What it does: Takes the user's question (like "How do I reset my password?") and converts it to the same type of vector as our stored documents.
        
        # Search in Chroma database
        results = self.collection.query(
            query_embeddings=[query_embedding], #Compares the query vector with all stored vectors
            n_results=n_results 
        )
        
        # Format results into a clean structure
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                }) #Finds the most similar ones (closest in meaning) and Returns the top 3 most relevant matches
        
        return formatted_results
    

    def generate_response(self, user_query: str) -> Dict:
        """
        Generate a response to user query using RAG with Claude Sonnet 4
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary containing the response and source information
        """
        # Step 1: Search knowledge base for relevant documents
        relevant_docs = self.search_knowledge_base(user_query, n_results=3) #What it does: Finds the 3 most relevant Q&A pairs from your knowledge base.
        
        if not relevant_docs:
            return {
                "response": "I'm sorry, I couldn't find relevant information to answer your question. Please contact our support team for further assistance.",
                "sources": [],
                "confidence": "low",
                "model_used": self.llm_model
            }
        
        # Step 2: Prepare context from retrieved documents
        context_sections = []
        sources = []
       
        #What it does: Takes the found documents and formats them nicely for Claude to read.
        for i, doc in enumerate(relevant_docs, 1):
            context_sections.append(f""" 
Source {i}:  
Category: {doc['metadata']['category']}
Question: {doc['metadata']['question']}
Answer: {doc['metadata']['answer']}
Relevance Score: {1 - doc['distance']:.3f}
""")
            
            sources.append({
                "category": doc['metadata']['category'],
                "question": doc['metadata']['question'],
                "distance": doc['distance'],
                "relevance_score": 1 - doc['distance']
            })
        
        context = "\n".join(context_sections)
        
        # Step 3: Create prompt for Claude Sonnet 4
        system_prompt = """You are an expert customer support assistant with access to a comprehensive knowledge base. Your role is to provide accurate, helpful, and professional responses to customer inquiries.

Guidelines:
- Use the provided context sources to answer questions accurately
- If the context doesn't fully address the question, clearly state what information is missing
- Be concise but thorough in your explanations
- Maintain a friendly, professional tone
- If multiple sources are relevant, synthesize the information effectively
- Always prioritize accuracy over completeness"""
        
        user_prompt = f"""Based on the following knowledge base sources, please answer the customer's question.

KNOWLEDGE BASE CONTEXT:
{context}

CUSTOMER QUESTION: {user_query}

Please provide a helpful and accurate response. If the available context doesn't contain sufficient information to fully answer the question, please indicate what additional information might be needed."""
        
        try:
            # Step 4: Generate response using Claude Sonnet 4, What it does: Sends everything to Claude Sonnet 4 and gets back a smart, contextual response.
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=500,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            ai_response = response.content[0].text
            
            # Step 5: Determine confidence based on similarity scores, What it does: Determines how confident we are in the answer based on how well the sources match the question.
            avg_distance = sum(doc['distance'] for doc in relevant_docs) / len(relevant_docs)
            confidence = "high" if avg_distance < 0.3 else "medium" if avg_distance < 0.6 else "low"
            
            return {
                "response": ai_response,
                "sources": sources,
                "confidence": confidence,
                "context_used": True,
                "model_used": self.llm_model,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
        except Exception as e:
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "sources": sources,
                "confidence": "error",
                "model_used": self.llm_model
            }
    #Complete Flow Example:

    #User asks: "How do I reset my password?"
    #Search finds: Password reset Q&A (distance: 0.2 = very similar)
    #Context built: Formats the Q&A for Claude
    #Claude Sonnet 4 reads: The context and user question
    #Claude responds: "To reset your password, go to the login page and click 'Forgot Password'..."
    #System returns: Response + confidence + sources used
import os
import re
import chromadb
import anthropic
import pandas as pd
from typing import List, Dict
import uuid
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# PDF reading support
import pdfplumber

load_dotenv()

# ---------------------------------------------------------------------------
# Keyword-extraction helpers — used by hybrid search in search_knowledge_base.
#
# Why these exist:
#   Embedding models (like all-MiniLM-L6-v2) are great at *concepts* but weak
#   at *exact tokens* — especially proper nouns (names, IDs, card numbers).
#   A question like "find Argneshu Gupta's name" may not semantically match
#   the chunk that literally contains "Argneshu Gupta", because the model has
#   no meaningful vector for a random person's name.
#
#   Solution: alongside the semantic search, also do a plain substring search
#   for every "meaningful" word in the user's query. That catches exact-token
#   lookups that embeddings miss.
# ---------------------------------------------------------------------------

# Common English words to EXCLUDE from keyword search. If we didn't filter
# these, queries like "what is the name on the card" would substring-match
# nearly every chunk (because "the", "is", "on" appear everywhere) and flood
# Claude's prompt with noise.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "of",
    "for", "by", "with", "as", "is", "are", "was", "were", "be", "been",
    "it", "this", "that", "these", "those", "what", "who", "where", "when",
    "why", "how", "can", "could", "would", "should", "do", "does", "did",
    "find", "show", "tell", "give", "please", "me", "my", "you", "your",
    "not", "no", "yes", "there", "their", "they", "we", "us", "our",
    # Words that describe *our system* rather than content inside chunks —
    # "pdf", "file", "document", "name" would also match almost anywhere.
    "pdf", "file", "document", "name", "names",
}


def _extract_keywords(query: str) -> List[str]:
    """
    Pull the content-bearing tokens out of a user query.

    Example:
        query = "Can you find Argneshu Gupta's name in the PDF?"
        returns -> ["Argneshu", "Gupta"]

    Rules:
      - Split on non-word characters (regex \\w+) so punctuation is dropped.
      - Keep only tokens of length >= 3 (drops "is", "on", "to", etc.).
      - Drop anything in _STOPWORDS (case-insensitive).
      - Deduplicate while preserving order (so the most important words first).
    """
    # Extract word-like tokens. Apostrophes are allowed so "Gupta's" still
    # tokenizes as "Gupta" and "s" (the "s" gets dropped by the length filter).
    tokens = re.findall(r"[A-Za-z0-9']+", query)

    seen, out = set(), []
    for t in tokens:
        key = t.lower()                              # normalize for dedupe + stopword check
        if len(t) < 3 or key in _STOPWORDS or key in seen:
            continue                                 # skip short, stop, or duplicate tokens
        seen.add(key)
        out.append(t)                                # keep ORIGINAL casing (matters for substring match)
    return out

class CustomerSupportRAG:
    def load_knowledge_base_from_pdf(
        self,
        pdf_file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        """
        Load a PDF into the knowledge base, split into SIZE-BASED chunks.

        Why size-based chunking (and not paragraph/line splitting):
          PDFs from different sources join lines in wildly different ways.
          Medical cards, invoices, and scanned forms often have NO double
          newlines, so splitting on "\\n\\n" collapses the whole PDF into
          one giant chunk — bad for retrieval (one chunk can't be ranked
          higher than another if there's only one).

          Fixed ~500-char windows with a small overlap guarantee:
            - Many chunks (so semantic search can actually rank them)
            - Context carried across chunk boundaries (via overlap),
              so facts like "Member Name : ARGNESHU GUPTA" aren't split
              in half between two chunks.

        Args:
            pdf_file_path: Path to the PDF file
            chunk_size:    Target size of each chunk in characters (default 500)
            chunk_overlap: Characters shared between consecutive chunks (default 100)
        """
        # Pull the full text out of the PDF (uses pdfplumber under the hood).
        text = self.read_pdf_text(pdf_file_path)
        if not text:
            print(f"❌ No text extracted from {pdf_file_path}")
            return

        # Build fixed-size chunks with overlap.
        # Example with chunk_size=500, overlap=100:
        #   chunk 0 covers chars [0 .. 500)
        #   chunk 1 covers chars [400 .. 900)   <- 100 char overlap with chunk 0
        #   chunk 2 covers chars [800 .. 1300)  <- 100 char overlap with chunk 1
        # The overlap prevents a key phrase from being cut in half.
        chunks: List[str] = []
        step = max(1, chunk_size - chunk_overlap)  # how far we advance each iteration
        for start in range(0, len(text), step):
            window = text[start:start + chunk_size].strip()
            if window:
                chunks.append(window)
            # Stop once the window has reached the end of the text.
            if start + chunk_size >= len(text):
                break

        # Build parallel lists of docs/metadata/ids for the Chroma add() call.
        documents = []
        metadatas = []
        ids = []
        # Use the file's basename + an index as ID prefix so re-ingesting a
        # different PDF doesn't collide with existing "pdf_0", "pdf_1" IDs.
        id_prefix = os.path.splitext(os.path.basename(pdf_file_path))[0]
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": pdf_file_path, "chunk_index": idx})
            ids.append(f"{id_prefix}_{idx}")

        print(f"Creating embeddings for {len(documents)} PDF chunks...")
        embeddings = self.create_embeddings_free(documents)
        # `upsert` instead of `add` so re-ingesting the same PDF refreshes
        # its chunks instead of erroring on duplicate IDs.
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"✅ Successfully loaded {len(documents)} PDF chunks into knowledge base from {pdf_file_path}")
    
    def read_pdf_text(self, pdf_file_path: str) -> str:
        """
        Extract all text from a PDF file using pdfplumber.
        Args:
            pdf_file_path: Path to the PDF file
        Returns:
            Extracted text as a single string
        """
        text = ""
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"❌ Error reading PDF: {str(e)}")
            return ""
    
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


    def search_knowledge_base(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        HYBRID search the knowledge base for relevant documents.

        Does TWO searches and merges their results:
          1) Semantic search  -> good for conceptual questions
             ("what is the insurance company?")
          2) Keyword search   -> good for exact tokens like proper nouns
             ("find Argneshu Gupta")

        Args:
            query: User's question
            n_results: Max number of semantic matches to retrieve

        Returns:
            List of relevant documents with metadata, deduplicated by doc text.
        """

        # ==============================================================
        # PART 1: SEMANTIC SEARCH (embedding-based similarity)
        # ==============================================================

        # Create embedding for the user's query
        query_embedding = self.create_embeddings_free([query])[0] #What it does: Takes the user's question (like "How do I reset my password?") and converts it to the same type of vector as our stored documents.

        # Search in Chroma database
        semantic = self.collection.query(
            query_embeddings=[query_embedding], #Compares the query vector with all stored vectors
            n_results=n_results
        )

        # Use a dict keyed by document text so we can merge semantic + keyword
        # hits later without creating duplicates. The doc text itself is the
        # dedupe key (two retrieval paths may surface the same chunk).
        results_by_doc: Dict[str, Dict] = {}

        # Format semantic results into a clean structure
        if semantic.get('documents') and semantic['documents'][0]:
            for i in range(len(semantic['documents'][0])):
                doc = semantic['documents'][0][i]
                results_by_doc[doc] = {
                    'document': doc,
                    'metadata': semantic['metadatas'][0][i],
                    'distance': semantic['distances'][0][i] if semantic.get('distances') else None
                } #Finds the most similar ones (closest in meaning) and keeps the top n_results most relevant matches

        # ==============================================================
        # PART 2: KEYWORD SEARCH (literal substring matching)
        # ==============================================================
        # Why:  embeddings miss proper nouns (names, IDs, codes). A substring
        #       scan guarantees that if the word literally exists in a chunk,
        #       that chunk gets retrieved.
        # How:  Chroma supports `where_document={"$contains": "..."}` which is
        #       a server-side LIKE-style filter. It's case-sensitive, so we try
        #       a few casings of each token (e.g. "Argneshu", "argneshu",
        #       "ARGNESHU") to be robust to how the PDF spelled it.

        # Pull meaningful tokens from the query (drops stopwords + short words).
        for token in _extract_keywords(query):

            # Try a few case variants because $contains in Chroma is case-sensitive.
            # Using a set prevents duplicate queries when the token is already
            # all lowercase (e.g. "argneshu" -> {"argneshu"} only).
            for variant in {token, token.lower(), token.capitalize(), token.upper()}:
                try:
                    # `collection.get` with where_document returns ALL chunks
                    # whose text contains the substring. `limit=5` prevents a
                    # common word from dragging in hundreds of chunks.
                    kw = self.collection.get(
                        where_document={"$contains": variant},
                        limit=5
                    )
                except Exception:
                    # If Chroma rejects the filter for any reason (bad chars,
                    # version mismatch, etc.), skip this variant gracefully
                    # rather than breaking the whole search.
                    continue

                # Merge any new hits into the results dict.
                for i, doc in enumerate(kw.get('documents') or []):
                    if doc in results_by_doc:
                        continue  # already retrieved semantically — skip
                    results_by_doc[doc] = {
                        'document': doc,
                        'metadata': (kw.get('metadatas') or [{}])[i],
                        # A literal substring match is essentially a perfect
                        # lexical hit; mark it with a low distance (0.1) so the
                        # confidence calculation in generate_response treats it
                        # as highly relevant.
                        'distance': 0.1
                    }

        # Return combined semantic + keyword hits as a flat list.
        return list(results_by_doc.values())
    

    def generate_response(self, user_query: str) -> Dict:
        """
        Generate a response to user query using RAG with Claude Sonnet 4
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary containing the response and source information
        """
        # Step 1: Search knowledge base for relevant documents
        relevant_docs = self.search_knowledge_base(user_query, n_results=10) #What it does: Finds the 3 most relevant Q&A pairs from your knowledge base.
        
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
            meta = doc['metadata']
            category = meta.get('category', '[no category]')

            # There are TWO kinds of chunks in this knowledge base:
            #   1) CSV rows   -> metadata has 'question' and 'answer' fields
            #   2) PDF chunks -> metadata has 'source' + 'chunk_index' only
            #
            # Previously, PDF chunks were rendered with a 60-char preview of
            # their text and a placeholder "[see document chunk]" — so Claude
            # never actually saw the PDF content. That's why questions like
            # "find the member name" kept failing even though the chunk was
            # retrieved correctly. Now we pass the FULL chunk text to Claude
            # whenever the row isn't a CSV Q&A.
            has_qa = 'question' in meta and 'answer' in meta
            if has_qa:
                question = meta['question']
                answer = meta['answer']
                context_sections.append(f"""
Source {i}:
Category: {category}
Question: {question}
Answer: {answer}
Relevance Score: {1 - doc['distance']:.3f}
""")
                source_question = question
            else:
                # PDF / free-text chunk — dump the raw document so Claude can
                # actually read the underlying content (e.g. "Member Name : X").
                source_label = meta.get('source', '[document]')
                context_sections.append(f"""
Source {i}:
Type: Document excerpt
Origin: {source_label}
Content:
{doc['document']}
Relevance Score: {1 - doc['distance']:.3f}
""")
                # For the UI's "View sources" panel, show a short preview.
                source_question = doc['document'][:80] + ('...' if len(doc['document']) > 80 else '')

            sources.append({
                "category": category,
                "question": source_question,
                "distance": doc['distance'],
                "relevance_score": 1 - doc['distance']
            })
        
        context = "\n".join(context_sections)
        
        # Step 3: Create prompt for Claude Sonnet 4
        system_prompt = """You are an expert customer support assistant with access to a comprehensive knowledge base. Your role is to provide accurate, helpful, and professional responses to customer inquiries.

The knowledge base contains two kinds of sources:
  1. "Q&A" style sources — curated customer support answers
  2. "Document excerpt" style sources — raw text pulled from uploaded documents
     (PDFs, etc.). When a source is a document excerpt, read the Content field
     carefully and extract specific facts the user is asking about (names,
     policy numbers, dates, etc.). Quote values directly from the excerpt.

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
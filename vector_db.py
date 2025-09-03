"""
Vector Database handler for long document processing
Uses ChromaDB for vector storage and sentence-transformers for embeddings
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qwen_agent.log import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logger.warning("Sentence transformers not installed. Install with: pip install sentence-transformers")

class VectorDatabase:
    def __init__(self, 
                 persist_directory: str = "./vector_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_cpu: bool = True):
        """
        Initialize vector database
        Args:
            persist_directory: Directory to persist the database
            embedding_model: Name of the sentence transformer model
            use_cpu: Use CPU for embeddings to save GPU memory
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.use_cpu = use_cpu
        
        if not CHROMADB_AVAILABLE or not SENTENCE_TRANSFORMER_AVAILABLE:
            self.enabled = False
            logger.warning("Vector database disabled. Install required packages.")
            return
        
        self.enabled = True
        
        # Initialize embedding model on CPU to save GPU memory
        device = 'cpu' if use_cpu else 'cuda'
        logger.info(f"Loading embedding model '{embedding_model}' on {device}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        # Initialize ChromaDB
        os.makedirs(persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collection for documents
        self.collection_name = "documents"
        self.collection = None
        
        logger.info(f"Vector database initialized at {persist_directory}")
    
    def _get_collection(self, recreate: bool = False):
        """Get or create collection"""
        if recreate and self.collection_name in [c.name for c in self.chroma_client.list_collections()]:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = None
        
        if self.collection is None:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        return self.collection
    
    def add_document(self, 
                    document_id: str,
                    chunks: List[str],
                    metadata: Optional[Dict[str, Any]] = None,
                    chunk_size: int = 512,
                    chunk_overlap: int = 128) -> bool:
        """
        Add document chunks to vector database
        Args:
            document_id: Unique identifier for the document
            chunks: List of text chunks
            metadata: Optional metadata for the document
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
        """
        if not self.enabled:
            return False
        
        try:
            collection = self._get_collection()
            
            # Process chunks if they're too large
            processed_chunks = []
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > chunk_size:
                    # Split large chunks
                    sub_chunks = self._split_text(chunk, chunk_size, chunk_overlap)
                    for j, sub_chunk in enumerate(sub_chunks):
                        processed_chunks.append(sub_chunk)
                        chunk_metadata.append({
                            "document_id": document_id,
                            "chunk_index": f"{i}_{j}",
                            "source": metadata.get("source", "unknown") if metadata else "unknown"
                        })
                else:
                    processed_chunks.append(chunk)
                    chunk_metadata.append({
                        "document_id": document_id,
                        "chunk_index": str(i),
                        "source": metadata.get("source", "unknown") if metadata else "unknown"
                    })
            
            if not processed_chunks:
                logger.warning(f"No chunks to add for document {document_id}")
                return False
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(processed_chunks)} chunks...")
            embeddings = self.embedding_model.encode(
                processed_chunks,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32
            )
            
            # Add to collection
            ids = [f"{document_id}_{meta['chunk_index']}" for meta in chunk_metadata]
            
            collection.add(
                embeddings=embeddings.tolist(),
                documents=processed_chunks,
                metadatas=chunk_metadata,
                ids=ids
            )
            
            logger.info(f"Added {len(processed_chunks)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to vector database: {e}")
            return False
    
    def search(self, 
              query: str, 
              n_results: int = 5,
              filter_document: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks
        Args:
            query: Search query
            n_results: Number of results to return
            filter_document: Optional document ID to filter results
        """
        if not self.enabled:
            return []
        
        try:
            collection = self._get_collection()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True
            )
            
            # Search with optional filter
            where_clause = {"document_id": filter_document} if filter_document else None
            
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results, collection.count()),
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0,
                        'id': results['ids'][0][i] if results['ids'] else ""
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        if not self.enabled:
            return
        
        try:
            self._get_collection(recreate=True)
            logger.info("Vector database collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def get_document_count(self) -> int:
        """Get number of chunks in the database"""
        if not self.enabled:
            return 0
        
        try:
            collection = self._get_collection()
            return collection.count()
        except:
            return 0
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # Only break if we have at least half the chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap if end < text_len else text_len
        
        return chunks
    
    def delete_document(self, document_id: str):
        """Delete all chunks for a specific document"""
        if not self.enabled:
            return
        
        try:
            collection = self._get_collection()
            collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted document {document_id} from vector database")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")


class RAGSearch:
    """RAG Search integration for qwen-agent"""
    
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        self.last_search_results = []
    
    def add_documents(self, docs: List[Any]) -> bool:
        """Add documents to vector database"""
        if not self.vector_db.enabled:
            logger.warning("Vector database not enabled")
            return False
        
        success = True
        for doc in docs:
            try:
                # Extract text content from document
                chunks = []
                doc_id = hashlib.md5(str(doc.url).encode()).hexdigest()
                
                for page in doc.raw:
                    if hasattr(page, 'content'):
                        chunks.append(page.content)
                
                if chunks:
                    success = self.vector_db.add_document(
                        document_id=doc_id,
                        chunks=chunks,
                        metadata={'source': doc.url}
                    ) and success
                    
            except Exception as e:
                logger.error(f"Error adding document to vector DB: {e}")
                success = False
        
        return success
    
    def search_relevant_chunks(self, query: str, max_chunks: int = 5) -> List[str]:
        """Search for relevant chunks based on query"""
        if not self.vector_db.enabled:
            return []
        
        results = self.vector_db.search(query, n_results=max_chunks)
        self.last_search_results = results
        
        # Return just the text chunks
        return [r['text'] for r in results]
    
    def get_context_for_query(self, query: str, max_tokens: int = 4000) -> str:
        """Get relevant context for a query within token limits"""
        if not self.vector_db.enabled:
            return ""
        
        # Search for more chunks than needed
        chunks = self.search_relevant_chunks(query, max_chunks=10)
        
        # Combine chunks up to token limit
        context = ""
        estimated_tokens = 0
        
        for chunk in chunks:
            # Rough token estimation (1 token ≈ 4 characters)
            chunk_tokens = len(chunk) // 4
            
            if estimated_tokens + chunk_tokens > max_tokens:
                break
            
            context += f"\n\n---\n\n{chunk}"
            estimated_tokens += chunk_tokens
        
        if context:
            context = f"Relevant context from documents:\n{context}\n\n---\n\nBased on the above context, "
        
        return context
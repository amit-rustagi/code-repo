import os
from typing import List, Dict, Any
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch

class RAGSystem:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530"):
        """
        Initialize RAG system with LLAMA model, embeddings, and Milvus vector store
        
        Args:
            model_name (str): Hugging Face LLAMA model name
            embedding_model (str): Embedding model for text vectorization
            milvus_host (str): Milvus vector database host
            milvus_port (str): Milvus vector database port
        """
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize LLAMA model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map='auto'
        )
        
        # Create text generation pipeline
        self.llm_pipeline = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer,
            max_length=1024
        )
        
        # Initialize LangChain LLM wrapper
        self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)
        
        # Connect to Milvus
        connections.connect(host=milvus_host, port=milvus_port)
        
        # Milvus collection configuration
        self.collection_name = "document_collection"
        self.create_milvus_collection()
    
    def create_milvus_collection(self):
        """Create Milvus vector collection for document storage"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        
        schema = CollectionSchema(fields, "RAG document collection")
        collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
    
    def load_documents(self, directory: str) -> List[Document]:
        """
        Load documents from a directory
        
        Args:
            directory (str): Path to document directory
        
        Returns:
            List of loaded documents
        """
        loader = DirectoryLoader(directory, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)
    
    def embed_and_store_documents(self, documents: List[Document]):
        """
        Embed documents and store in Milvus
        
        Args:
            documents (List[Document]): Documents to embed and store
        """
        collection = Collection(self.collection_name)
        
        for doc in documents:
            # Generate embedding
            embedding = self.embeddings.embed_query(doc.page_content)
            
            # Insert document and embedding
            collection.insert([
                doc.page_content,
                embedding
            ])
        
        collection.flush()
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant documents for a query
        
        Args:
            query (str): Search query
            top_k (int): Number of top documents to retrieve
        
        Returns:
            List of relevant document texts
        """
        collection = Collection(self.collection_name)
        
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in Milvus
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        results = collection.search(
            [query_embedding], 
            anns_field="embedding", 
            param=search_params, 
            limit=top_k, 
            output_fields=["text"]
        )
        
        return [hit.entity.get('text') for hit in results[0]]
    
    def generate_response(self, query: str) -> str:
        """
        Generate RAG response by retrieving and augmenting context
        
        Args:
            query (str): User query
        
        Returns:
            Generated response
        """
        # Retrieve relevant documents
        context_docs = self.retrieve_relevant_docs(query)
        
        # Construct augmented prompt
        augmented_prompt = f"""
        Context: {' '.join(context_docs)}
        
        Question: {query}
        
        Based on the context, provide a detailed and accurate response:
        """
        
        # Generate response using LLAMA
        response = self.llm(augmented_prompt)
        return response
    
    def index_documents(self, directory: str):
        """
        Full document indexing pipeline
        
        Args:
            directory (str): Path to documents directory
        """
        documents = self.load_documents(directory)
        self.embed_and_store_documents(documents)

def main():
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Index documents from a directory
    rag_system.index_documents("./documents")
    
    # Example query
    query = "What are the key principles of machine learning?"
    response = rag_system.generate_response(query)
    print("Query:", query)
    print("Response:", response)

if __name__ == "__main__":
    main()

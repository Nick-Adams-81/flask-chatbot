from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional
import numpy as np
from langchain_openai import OpenAIEmbeddings

class PineconeVectorDB:
    def __init__(self, api_key: str, environment: str, index_name: str = "tda-rules"):
        """
            Initialize Pinecone vector database.

        Args:
            api_key: Pinecone api key
            environment: Pinecone environment(e.g. us-east1-gcp)
            index_name: Name of index
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index = None
        
        pc = Pinecone(api_key=api_key)
        self.pc = pc  # Store as instance variable

        # Create index if it doesnt exist
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name, 
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        # Connect to index
        self.index = pc.Index(self.index_name)

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of dictionaries containing 'text' and 'metadata'
        """
        vectors = []

        for doc in documents:
            embedding = self.embedding_model.embed_query(doc["text"])

            # Create vector record
            vector_record = {
                'id': doc.get('id', f"doc_{len(vectors)}"),
                'values': embedding,
                'metadata': {
                    'text': doc['text'],
                    **doc.get('metadata', {})
                }
            }
            vectors.append(vector_record)

        # Upsert vectors to Pinecone
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search the vector database for the most relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Dictionary of filters

        Returns:
            List of dictionaries containing 'text' and 'metadata'
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )

        return [
            {
                'text': match.metadata['text'],
                'score': match.score,
                'metadata': match.metadata
            }
            for match in results.matches
        ]

    def delete_index(self) -> None:
        """Delete the index (use with caution)"""
        self.pc.delete_index(self.index_name)
            
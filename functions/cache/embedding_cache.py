import numpy as np
import hashlib
from typing import Dict, Optional, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

class EmbeddingCache:
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """
        Initialize the embedding cache.

        Args:
            max_size: Maximum number of embeddings to store.
            ttl_hours: Time-to-live for cached embeddings (Default: 24 hours)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600

        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # Storage
        self.cache: Dict[str, Dict] = {}
        self.access_order: list = []

        # Stats
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.api_calls_saved = 0
        self.creation_time = datetime.now()

    def generate_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text"""
        # normalize text: lowercase, strip whitespace, normalize spaces
        normalized_text = ' '.join(text.lower().strip().split())
        return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for given text using OpenAI."""
        try:
            embedding = self.embedding_model.embed_query(text)
            return np.array(embedding).reshape(1, -1)
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return None

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text, using cache if available.

        Args:
            text: The text to get embedding for.

        Returns:
            Embedding for the text, or None if failed.
        """
        self.total_requests += 1
        cache_key = self.generate_cache_key(text)

        # Check if embedding exists in cache and is not expired
        if cache_key in self.cache:
            entry = self.cache[cache_key]

            # Check ttl
            age_seconds = (datetime.now() - entry['timestamp']).total_seconds()
            if age_seconds < self.ttl_seconds:
                # cache hit - update access info
                self.hits += 1
                entry['access_count'] += 1
                entry['last_accessed'] = datetime.now()

                # update access order for LRU
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
                self.access_order.append(cache_key)

                print(f"Embedding cache hit for: {text[:50]}...")
                return entry['embedding']
            else:
                # Expired - remove from cache
                del self.cache[cache_key]
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
        # Cache miss - generate new embedding
        self.misses += 1
        print(f"Embedding cache MISS for: {text[:50]}...")

        embedding = self.generate_embedding(text)
        if embedding is not None:
            # Check if cache is full and evict if necessary
            if len(self.cache) >= self.max_size:
                self.evict_oldest()

            # Store in cache
            cache_entry = {
                'embedding': embedding,
                'original_text': text,
                'timestamp': datetime.now(),
                'last_accessed': datetime.now(),
                'access_count': 1
            }

            self.cache[cache_key] = cache_entry
            self.access_order.append(cache_key)

            print(f"Embedding cached for: {text[:50]}...")
        return embedding

    def evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if self.access_order:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
                print(f"Evicted embedding from cache: {self.cache[oldest_key]['original_text'][:50]}...")

    def get_stats(self) -> Dict:
        """Get cache stats."""
        hit_rate = self.hits / max(self.total_requests, 1)
        api_calls_saved = self.hits
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'hit_rate': hit_rate,
            'api_calls_saved': self.api_calls_saved,
            'creation_time': self.creation_time,
            'ttl_hours': self.ttl_seconds / 3600
        }

    def clear_cache(self) -> None:
        """Clear the cache"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.api_calls_saved = 0
        print("Embedding cache cleared")

    def get_most_accessed(self, limit: int = 10) -> list:
        """Get most accessed embeddings"""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x : x[1]['access_count'],
            reverse=True
        )
        return [
            {
                'text': entry[1]['original_text'][:100] + '...' if len(entry[1]['original_text']) > 100 else entry[1]['original_text'],
                'access_count': entry[1]['access_count'],
                'last_accessed': entry[1]['last_accessed']
            }
            for entry in sorted_entries[:limit]
        ]

    def find_similar_cached_embedding(self, text: str, similarity_threshold: float = 0.95) -> Optional[np.ndarray]:
        """
        Find a cached embedding for semantically similar text.
        
        Args:
            text: The text to find similar embedding for
            similarity_threshold: Minimum similarity score
            
        Returns:
            Similar embedding or None if not found
        """

        input_embedding = self.generate_embedding(text)
        if input_embedding is None:
            return None
        
        # Check all cached embeddings for similarity
        for cache_key, entry in self.cache.items():
            cached_embedding = entry['embedding']
            similarity = cosine_similarity(input_embedding, cached_embedding)[0][0]
            
            if similarity >= similarity_threshold:
                print(f"Found similar cached embedding (similarity: {similarity:.3f})")
                return cached_embedding
        
        return None
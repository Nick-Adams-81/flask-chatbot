from smtplib import OLDSTYLE_AUTH
import numpy as np
from datetime import date, datetime
import hashlib
from typing import Dict, Optional, Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

class Cache:
    def __init__(self, max_cache_size: int = 1000, similarity_threshold: float = 0.85, eviction_policy: str = "lru"):
        """
        Initialize the cache with in-memory dictionary storage.

        Args:
            max_cache_size: Maximum number of items to store in the cache.
            similarity_threshold: Minimum cosine similarity for cache hits.
            eviction_policy: "lru", "similarity", or "hybrid"
        """
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold
        self.eviction_policy = eviction_policy

        # Storage
        self.cache: Dict[str, Dict] = {}
        self.access_order: List[str] = []

        # Embedding model
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # Stats
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0
        self.creation_time = datetime.now()

    def generate_embedding(self, text:str) -> np.ndarray:
        """Generate embedding for given text"""
        try:
            embedding = self.embedding_model.embed_query(text)
            return np.array(embedding).reshape(1, -1)
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return None
        
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0

    def get_embedding_hash(self, embedding: np.ndarray) -> str:
        """Generate hash for embedding to use as cache key"""
        return hashlib.md5(embedding.tobytes()).hexdigest()

    def find_similar_question(self, question_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find the most similar cached question and its similarity score"""
        best_match = None
        best_similarity = 0.0

        for cache_key, entry in self.cache.items():
            similarity = self.calculate_similarity(question_embedding, entry["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cache_key

        return best_match, best_similarity

    def get_response(self, question: str) -> Optional[str]:
        """Main method: check cache for similar question and return response if found"""
        self.total_requests += 1

        # Generate embedding for the question
        question_embedding = self.generate_embedding(question)
        if question_embedding is None:
            self.miss_count += 1
            return None

        # Find similar question in cache
        best_match_key, similarity_score = self.find_similar_question(question_embedding)
        
        # DEBUG: Print similarity information
        print(f"Question: '{question}'")
        print(f"Best similarity score: {similarity_score:.3f}")
        print(f"Threshold: {self.similarity_threshold}")
        if best_match_key:
            print(f"Best match: '{self.cache[best_match_key]['original_question']}'")
        else:
            print("No similar questions found in cache")

        # Check if similarity is above threshold
        if best_match_key and similarity_score >= self.similarity_threshold:
            # Cache hit - update stats and access info
            self.hit_count += 1
            self.cache[best_match_key]["access_count"] += 1
            self.cache[best_match_key]["last_accessed"] = datetime.now()

            # Update access order in LRU cache
            if best_match_key in self.access_order:
                self.access_order.remove(best_match_key)
            self.access_order.append(best_match_key)

            print(f"CACHE HIT! Returning cached response")
            return self.cache[best_match_key]["response"]
        else:
            # Cache miss
            self.miss_count += 1
            print(f"CACHE MISS! Similarity {similarity_score:.3f} < threshold {self.similarity_threshold}")
            return None

    def add_response(self, question: str, response: str) -> None:
        """Store new question-response pair in cache"""
        # Generate embedding
        question_embedding = self.generate_embedding(question)
        if question_embedding is None:
            return

        # Check if cache is full and evict if necessary
        if len(self.cache) >= self.max_cache_size:
            self.evict_entries()
        
        # Create cache entry
        cache_key = self.get_embedding_hash(question_embedding)
        cache_entry = {
            "embedding": question_embedding,
            "original_question": question,
            "response": response,
            "timestamp": datetime.now(),
            "access_count": 0,
            "similarity_score": 1.0, # Perfect match for self
            "embedding_hash": cache_key
        }

        # Store in cache
        self.cache[cache_key] = cache_entry
        self.access_order.append(cache_key)

    def evict_entries(self) -> None:
        """Evict entries based on eviction policy"""
        if len(self.cache) < self.max_cache_size:
            return

        entries_to_remove = len(self.cache) - self.max_cache_size + 1

        if self.eviction_policy == "lru":
            # Remove least recently used entries
            for _ in range(entries_to_remove):
                if self.access_order:
                    oldest_key = self.access_order.pop(0)
                    del self.cache[oldest_key]

        elif self.eviction_policy == "similarity":
            # Remove entries with lowest similarity scores
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1]["similarity_score"]
            )
            for i in range(entries_to_remove):
                if i < len(sorted_entries):
                    key_to_remove = sorted_entries[i][0]
                    del self.cache[key_to_remove]
                    if key_to_remove in self.access_order:
                        self.access_order.remove(key_to_remove)

        elif self.eviction_policy == "hybrid":
            # Combine access count and similarity
            for key, entry in self.cache.items():
                days_since_creation = (datetime.now() - entry["timestamp"]).days
                if days_since_creation == 0:
                    days_since_creation = 1
                entry["hybrid_score"] = (entry["access_count"] * entry["similarity_score"]) / days_since_creation

            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1]["hybrid_score"]
            )
            for i in range(entries_to_remove):
                if i < len(sorted_entries):
                    key_to_remove = sorted_entries[i][0]
                    del self.cache[key_to_remove]
                    if key_to_remove in self.access_order:
                        self.access_order.remove(key_to_remove)

    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache stats"""
        return {
            "total_entries": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_count / max(self.total_requests, 1),
            "creation_time": self.creation_time,
            "similarity_threshold": self.similarity_threshold,
            "eviction_policy": self.eviction_policy
        }

    def get_hit_rate(self) -> float:
        """Calculate cache hit percentage"""
        return self.hit_count / max(self.total_requests, 1)

    def clear_cache(self) -> None:
        """Clear the cache of all entries"""
        self.cache.clear()
        self.access_order.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0

    def get_cache_size(self) -> int:
        """Return current number of cache entries"""
        return len(self.cache)

    def set_similarity_threshold(self, threshold: float) -> None:
        """Update similarity threshold"""
        self.similarity_threshold = threshold

    def set_max_cache_size(self, size: int) -> None:
        """Update max cache size"""
        self.max_cache_size = size
        if len(self.cache) > size:
            self.evict_entries()

    def get_most_accessed_entries(self, limit: int = 10) -> List[Dict]:
        """Get most frequently accessed entries"""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]["access_count"],
            reverse=True
        )
        return [{"question": entry[1]["original_question"],
                "access_count": entry[1]["access_count"]}
                for entry in sorted_entries[:limit]
        ]
    
    def validate_cache_integrity(self) -> bool:
        """Check if all cache entries are valid"""
        for key, entry in self.cache.items():
            required_fields = ["embedding", "original_question", "response",
                             "timestamp", "access_count", "similarity_score"]
            if not all(field in entry for field in required_fields):
                return False
        return True
    
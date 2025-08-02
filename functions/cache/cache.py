import numpy as np
from datetime import datetime
import hashlib
from typing import Dict, Optional, Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from .embedding_cache import EmbeddingCache
import re

class Cache:
    def __init__(self, max_cache_size: int = 1000, similarity_threshold: float = 0.75, eviction_policy: str = "lru"):
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

        # Embedding cache
        self.embedding_cache = EmbeddingCache(max_size=1000, ttl_hours=24)

        # Storage
        self.cache: Dict[str, Dict] = {}
        self.access_order: List[str] = []

        # Embedding model
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # Single compiled regex pattern for number extraction
        self.number_pattern = re.compile(r"""
            (?P<players>(\d+)\s*(?:players?|people|participants?))|
            (?P<tables>(\d+)\s*tables?|table\s*(\d+))|
            (?P<chips>(\d+)\s*(?:chips?|stack))|
            (?P<blinds>(\d+)\s*blinds?|blind\s*(\d+))|
            (?P<cards>(\d+)\s*cards?)|
            (?P<general>(\d+))
        """, re.IGNORECASE | re.VERBOSE)

        # Stats
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0
        self.creation_time = datetime.now()

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text"""
    
        embedding = self.embedding_cache.get_embedding(text)
        if embedding is not None:
            return embedding
        else:
            print(f"Failed to generate embedding for: {text[:50]}...")
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

    def extract_numbers_with_context(self, text: str) -> Dict[str, List[int]]:
        """Extract numbers from text and group them by context.

        Args: 
           text: The input text to exstract numbers

        Returns:
            Dictionary with context as key and list of numbers as value
            Ex: {"players": [8, 6], "tables": [2], "chips":[1000]}
        """
        result = {}

        # Single pass through text
        for match in self.number_pattern.finditer(text):
            # extract the context from the named group
            context = match.lastgroup

            # extract the number from the appropriate group
            number = None
            if context == "players":
                number = match.group(2)
            elif context == "tables":
                # check which pattern matched for tables
                if match.group(3):
                    number = match.group(3)
                else:
                    number = match.group(4)
            elif context == "chips":
                number = match.group(5)
            elif context == "blinds":
                if match.group(6):
                    number = match.group(6)
                else:
                    number = match.group(7)
            elif context == "cards":
                number = match.group(8)
            elif context == "general":
                number = match.group(9)
            else:
                continue

            # Only add if we successfully extracted a number
            if number is not None:
                try:
                    number_int = int(number)
                    if context not in result:
                        result[context] = []
                    result[context].append(number_int)
                except (ValueError, TypeError):
                    # Skip invalid numbers
                    continue

        # sort numbers for consistent comparison
        for context in result:
            result[context] = sorted(result[context])

        return result

    def compare_number_patterns(self, numbers1: Dict[str, List[int]], numbers2: Dict[str, List[int]]) -> float:
        """Compare number patterns between two questions and return similarity score
        
        Args:
            numbers1: Numbers from first question
            numbers2: Numbers from second question

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not numbers1 and not numbers2:
            return 1.0
        
        if not numbers1 or not numbers2:
            return 0.6
        
        all_contexts = set(numbers1.keys()) | set(numbers2.keys())
        total_similarity = 0.0
        total_contexts = 0

        for context in all_contexts:
            nums1 = numbers1.get(context, [])
            nums2 = numbers2.get(context, [])

            if nums1 == nums2:
                total_similarity += 1.0
            elif nums1 and nums2:
                if len(nums1) == len(nums2):
                    differences = [abs(a - b) for a, b in zip(nums1, nums2)]
                    avg_difference = sum(differences) / len(differences)
                    max_value = max(max(nums1), max(nums2))

                    # Much stricter number comparison - any difference should heavily penalize similarity
                    if avg_difference == 0:
                        total_similarity += 1.0
                    elif avg_difference <= 1:
                        total_similarity += 0.1  # Reduced from 0.3
                    elif avg_difference <= max_value * 0.05:  # Reduced from 0.1
                        total_similarity += 0.05  # Reduced from 0.1
                    else:
                        total_similarity += 0.0
                else:
                    total_similarity += 0.05  # Reduced from 0.1
            else:
                total_similarity += 0.2  # Reduced from 0.3
            
            total_contexts += 1
        return total_similarity / total_contexts if total_contexts > 0 else 1.0

    def calculate_number_aware_similarity(self, question1: str, question2: str, embedding_similarity: float) -> float:
        """
        Calculate similarity that takes into account both semantic similarity and number patterns.
        
        Args:
            question1: First question
            question2: Second question
            embedding_similarity: Semantic similarity from embeddings
            
        Returns:
            Combined similarity score
        """
        # Extract numbers from both questions
        numbers1 = self.extract_numbers_with_context(question1)
        numbers2 = self.extract_numbers_with_context(question2)
        
        # Calculate number similarity
        number_similarity = self.compare_number_patterns(numbers1, numbers2)
        
        # Combine embedding similarity with number similarity
        # Weight: 50% embedding similarity, 50% number similarity (increased number weight)
        final_similarity = (embedding_similarity * 0.5) + (number_similarity * 0.5)
        
        # Debug logging
        print(f"Embedding similarity: {embedding_similarity:.3f}")
        print(f"Number similarity: {number_similarity:.3f}")
        print(f"Final similarity: {final_similarity:.3f}")
        print(f"Numbers1: {numbers1}")
        print(f"Numbers2: {numbers2}")
        
        return final_similarity

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
        # Step 1: Check for EXACT text match first (response cache)
        if question in self.cache:
            print(f"Exact match found for: {question}")
            self.hit_count += 1
            self.total_requests += 1

            #update access order for LRU
            if question in self.access_order:
                self.access_order.remove(question)
            self.access_order.append(question)

            #return cached response
            return self.cache[question]["response"]
        
        # Step 2: Generate embedding for the question (this will use embedding cache if available)
        question_embedding = self.generate_embedding(question)
        if question_embedding is None:
            self.miss_count += 1
            return None

        # Find similar question in cache
        best_match_key, embedding_similarity = self.find_similar_question(question_embedding)
        
        # DEBUG: Print similarity information for ALL requests
        print(f"Question: '{question}'")
        print(f"Best embedding similarity score: {embedding_similarity:.3f}")
        print(f"Threshold: {self.similarity_threshold}")
        if best_match_key:
            print(f"Best match: '{self.cache[best_match_key]['original_question']}'")
        else:
            print("No similar questions found in cache")

        # Apply number-aware similarity adjustment
        if best_match_key and embedding_similarity >= self.similarity_threshold:
            # Calculate number-aware similarity
            cached_question = self.cache[best_match_key]['original_question']
            final_similarity = self.calculate_number_aware_similarity(question, cached_question, embedding_similarity)
            
            # Check if final similarity is above threshold
            if final_similarity >= self.similarity_threshold:
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
                # Number similarity brought it below threshold
                self.miss_count += 1
                print(f"CACHE MISS! Final similarity {final_similarity:.3f} < threshold {self.similarity_threshold}")
                return None
        else:
            # Cache miss
            self.miss_count += 1
            print(f"CACHE MISS! Embedding similarity {embedding_similarity:.3f} < threshold {self.similarity_threshold}")
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
    
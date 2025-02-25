import numpy as np
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Initialize embeddings model once
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Reference text for relevance checking
REFERENCE_TEXT = "Tournament Directors Association (TDA) rules for poker tournaments."
REFERENCE_EMBEDDING = np.array(embeddings.embed_query(REFERENCE_TEXT)).reshape(1, -1)

# Predefined keyword list
ALLOWED_KEYWORDS = ["poker", "tournament", "rules", "blinds", "chips", "floor", "dealer", "hand rankings"]

def get_embedding(text):
    """Converts text to an embedding vector."""
    return np.array(embeddings.embed_query(text)).reshape(1, -1)

def extract_keywords_from_text(text):
    """Extracts significant keywords from the source text based on frequency and simple word filtering."""
    # Use regex to extract words (ignoring common stop words)
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Define simple stop words to ignore
    stop_words = {"the", "and", "for", "to", "a", "an", "is", "in", "on", "of", "with"}
    
    # Remove stop words and return filtered list of keywords
    filtered_words = [word for word in words if word not in stop_words]
    
    # Return the top keywords by frequency (you can adjust the logic as needed)
    return filtered_words

# Dynamically extract keywords from the source material
def get_allowed_keywords(document_text):
    return extract_keywords_from_text(document_text)

def is_question_relevant(user_input, allowed_keywords, reference_embedding):
    """Checks if a question is related to TDA rules using keywords & embeddings."""
    # Keyword check
    if any(keyword in user_input.lower() for keyword in allowed_keywords):
        return True

    # Embedding similarity check
    question_embedding = get_embedding(user_input)
    similarity_score = cosine_similarity(question_embedding, reference_embedding)[0][0]

    return similarity_score > 0.7
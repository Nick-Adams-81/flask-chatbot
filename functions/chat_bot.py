import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from .guardrails.context_guardrail import get_allowed_keywords, is_question_relevant, REFERENCE_TEXT, get_embedding, get_reference_embedding
from .guardrails.safety_guardrail import safety_guardrail
from .data_ingestion.txt_loader import txt_loader, retriever
from .cache.cache import Cache
from .vector_db.pinecone_db import PineconeVectorDB

load_dotenv()

def validate_environment():
    """Validate required environment variables."""
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {missing_vars}")
        print("Please check your .env file")
        return False
    
    return True

# Validate environment variables
if not validate_environment():
    raise ValueError("Missing required environment variables")

api_key = os.getenv("OPENAI_API_KEY")

# Initialize chat history globally (or you can use a database/session for persisting)
chat_history = []

cache_manager = Cache(max_cache_size=100, similarity_threshold=0.60, eviction_policy="lru")

# Initialize Pinecone with error handling
try:
    pinecone_db = PineconeVectorDB(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
        index_name="tda-rules"
    )
    print("Pinecone initialized successfully")
    pinecone_available = True
except ValueError as e:
    print(f"Pinecone configuration error: {e}")
    print("Please check your PINECONE_API_KEY and PINECONE_ENVIRONMENT")
    pinecone_db = None
    pinecone_available = False
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    pinecone_db = None
    pinecone_available = False

def fallback_search(user_input: str, document_path: str):
    """Fallback search using original FAISS method."""
    try:
        document = txt_loader(document_path)
        retrieve = retriever(document)
        
        # Get relevant documents using FAISS
        docs = retrieve.get_relevant_documents(user_input)
        
        # Format for consistency with Pinecone results
        return [
            {
                'text': doc.page_content,
                'score': 0.8,  # Default score for fallback
                'metadata': doc.metadata
            }
            for doc in docs
        ]
    except Exception as e:
        print(f"Fallback search also failed: {e}")
        return []

def load_documents_to_pinecone(document_path: str):
    """
    Load and chunk documents to Pinecone vector database.

    Args:
        document_path: Path to the document to load
    """
    if not pinecone_available:
        print("Pinecone not available, skipping document loading")
        return False
        
    try:
        # Check if file exists
        if not os.path.exists(document_path):
            print(f"Error: Document file not found at {document_path}")
            return False
            
        # Read the document
        with open(document_path, "r") as file:
            content = file.read()

        # Split into chunks
        chunk_size = 1000
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

        # Create documents for Pinecone
        documents = [
            {
                'text': chunk,
                'metadata': {
                    'source': 'tournament-rules.txt',
                    'chunk_id': i,
                    'chunk_size': len(chunk)
                }
            }
            for i, chunk in enumerate(chunks)
        ]

        # Add documents to Pinecone
        pinecone_db.add_documents(documents)
        print(f"Loaded {len(documents)} documents into Pinecone")
        return True
        
    except FileNotFoundError:
        print(f"Error: Document file not found at {document_path}")
        return False
    except Exception as e:
        print(f"Error loading documents to Pinecone: {e}")
        return False
    
def chat_bot(document_path, user_input):
    # Ensure chat_history is global so it persists across calls
    global chat_history

    # Check if the user input contains harmful language using the safety guardrail
    is_profane, profane_words = safety_guardrail(user_input)
    if is_profane:
        return f"Please refrain from using harmful or offensive language. Detected words: {', '.join(profane_words)}"

    # Check cache first
    cached_response = cache_manager.get_response(user_input)
    print(f"Cache lookup for: {user_input}")
    print(f"Cached response: {cached_response}")
    # Cache hit - return cached response
    if cached_response:
        return cached_response

    # Cache miss - proceed with normal processing
    # Using our LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Use pinecone for document retrieval with error handling and fallback
    try:
        if pinecone_available:
            print(f"Searching Pinecone for: {user_input}")
            similar_docs = pinecone_db.search(user_input, top_k=3)
            print(f"Pinecone returned {len(similar_docs)} documents")
            if similar_docs:
                print(f"First document: {similar_docs[0]['text'][:100]}...")
            if not similar_docs:
                print("Warning: No similar documents found in Pinecone, using fallback")
                similar_docs = fallback_search(user_input, document_path)
        else:
            print("Pinecone not available, using fallback search")
            similar_docs = fallback_search(user_input, document_path)
            
    except Exception as e:
        print(f"Error searching Pinecone: {e}, using fallback")
        similar_docs = fallback_search(user_input, document_path)
        
        if not similar_docs:
            return "I'm having trouble accessing the rules database. Please try again later."

    # Format docs for the LLM
    context_text = "\n\n".join([doc['text'] for doc in similar_docs])

    # Create a simple document-like structure for the chain
    documents = [Document(page_content=context_text, metadata={})]

    # Read the document with error handling
    try:
        with open(document_path, "r") as file:
            document_text = file.read()
    except FileNotFoundError:
        return "Error: Rules document not found. Please contact support."
    except Exception as e:
        print(f"Error reading document: {e}")
        return "I'm having trouble reading the rules. Please try again later."

    # Get allowed keywords for relevance check
    allowed_keywords = get_allowed_keywords(document_text)
    reference_embedding = get_reference_embedding()

    # Check if the user's question is relevant using the guardrail
    if not is_question_relevant(user_input, allowed_keywords, reference_embedding):
        return "I can only answer questions related to TDA rules and rulings."

    # Add the user input to the chat history
    chat_history.append({"user": user_input})

    # Format the chat history for the prompt
    history_text = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in chat_history if 'bot' in msg])

    # Prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """ You are an assistant for answering questions about 
        tournament poker rules. Use the provided context to respond. If the user query is not relevant,
        say you can't answer. If the answer isn't clear, acknowledge that you don't know. 
        Limit your response to three concise sentences.
        
        Context: {context}
        """),
        ("human", "{history}\nUser: {input}")
    ])

    # Create chain - use simple LLM chain instead of document chain
    from langchain.chains import LLMChain
    qa_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Use documents directly instead of retrieval chain
    try:
        response = qa_chain.invoke({
            "input": user_input,
            "context": context_text,
            "history": history_text
        })
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm having trouble generating a response. Please try again later."

    # Extract bot's answer
    bot_answer = response['text'] # LLMChain returns a dict with 'text' key

    # Cache the question-response pair
    cache_manager.add_response(user_input, bot_answer)
    print(f"Caching response for: {user_input}")
    print(f"Cache size: {cache_manager.get_cache_size()}")

    # Append bot response to chat history
    # Add the bot's response to the latest entry
    chat_history[-1]["bot"] = bot_answer

    return bot_answer

def initialize_pinecone():
    """Initialize Pinecone vector database."""
    document_path = "./data/tournament-rules.txt"
    success = load_documents_to_pinecone(document_path)
    if success:
        print("Pinecone initialization completed successfully")
    else:
        print("Pinecone initialization failed")
    return success

#initialize_pinecone()

import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from dotenv import load_dotenv
from functions import cache
from functions.chat_bot import chat_bot
from functions.cache.cache import Cache

# # Load document path
DOCUMENT_PATH = "./data/tournament-rules.txt"

# Load environment variables
load_dotenv()

app = Flask(__name__)
# For session management
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")

cache_manager = Cache(max_cache_size=1000, similarity_threshold=0.75, eviction_policy="lru")

def pre_warm_cache():
    """Pre-warm the cache with common questions and their responses."""
    common_questions = [
        'what is the rule on all ins?',
        'what are the blind rules?',
        'how do chips work in tournaments?',
        'what happens when someone is all-in?',
        'what are the hand rankings?',
        'what is the rule on string betting?',
        'how do side pots work?',
        'what is the rule on showing cards?',
        'what happens if a player acts out of turn?',
        'what are the rules for dealers?'
    ]
    
    print("Pre-warming cache with common questions and responses...")
    for question in common_questions:
        try:
            # Step 1: Generate embedding (existing)
            cache_manager.generate_embedding(question)
            
            # Step 2: Generate response using the chat bot
            response = chat_bot("./data/tournament-rules.txt", question)
            
            # Step 3: Add to response cache
            cache_manager.add_response(question, response)
            
            print(f"Pre-warmed: {question}")
            print(f"Response: {response[:100]}...")
            
        except Exception as e:
            print(f"Failed to pre-warm: {question} - {e}")
    
    print("Cache pre-warming completed!")

# Call pre-warming at startup
pre_warm_cache()

# Fetch credentials from .env
USERNAME = os.getenv("ADMIN_USERNAME")
PASSWORD = os.getenv("ADMIN_PASSWORD")

# Health check endpoint
@app.route("/health")
def health():
    return jsonify({"status": "healthy", "message": "Flask app is running"}), 200

# Login route
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == USERNAME and password == PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("chat"))  # Redirect to chatbot page
        else:
            return "Invalid credentials, please try again."

    return render_template("login.html")  # Render login page

# Logout route
@app.route("/logout")
def logout():
    session.pop("authenticated", None)
    return redirect(url_for("login"))

# Chatbot route (protected)
@app.route("/chat")
def chat():
    if not session.get("authenticated"):
        return redirect(url_for("login"))
    return render_template("index.html")

# API route (protected)
@app.route("/chatbot", methods=["POST"])
def chatbot():
    if not session.get("authenticated"):
        return jsonify({"response": "Unauthorized"}), 401

    data = request.get_json()

    # Check if "question" key is in the request
    if not data or "question" not in data:
        return jsonify({"response": "Invalid request"}), 400

    user_input = data["question"]
    
    # Bot response
    bot_response = chat_bot(DOCUMENT_PATH, user_input)

    return jsonify({"response": bot_response})

@app.route("/cache-stats")
def cache_stats():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    
    stats = cache_manager.get_cache_stats()
    embedding_stats = cache_manager.embedding_cache.get_stats()
    combined_stats = {
        "response_cache": stats,
        "embedding_cache": embedding_stats,
        "total_api_calls_saved": stats.get("api_calls_saved", 0) + embedding_stats.get("api_calls_saved", 0)

    }
    return jsonify(combined_stats)

if __name__ == "__main__":
    # Use Heroku's PORT environment variable, fallback to 6500 for local development
    port = int(os.environ.get("PORT", 6500))
    # Disable debug mode in production
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=debug, host="0.0.0.0", port=port)

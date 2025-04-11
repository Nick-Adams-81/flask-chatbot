import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from dotenv import load_dotenv
from functions.chat_bot import chat_bot

# # Load document path
DOCUMENT_PATH = "./data/tournament-rules.txt"

# Load environment variables
load_dotenv()

app = Flask(__name__)
# For session management
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")

# Fetch credentials from .env
USERNAME = os.getenv("ADMIN_USERNAME")
PASSWORD = os.getenv("ADMIN_PASSWORD")

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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6500)
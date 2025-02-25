from flask import Flask, render_template, request, jsonify
from functions.chat_bot import chat_bot

app = Flask(__name__)

# Load document path
DOCUMENT_PATH = ".venv/data/tournament-rules.txt"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("question", "")
    response = chat_bot(DOCUMENT_PATH, user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6500)

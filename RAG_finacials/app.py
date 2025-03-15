import os
import socket
os.system('pip install flask flask-cors')
os.system('pip install --upgrade pip')

import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from generator import generate_response

app = Flask(__name__, template_folder="templates")
CORS(app)  # Allow cross-origin requests if your frontend is separate

# 1️⃣ Serve a basic front-end page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# 2️⃣ API Route
@app.route("/query", methods=["POST"])
def handle_query():
    """Receives a user query, retrieves relevant docs, generates a RAG-based response."""
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        # Generate response
        answer = generate_response(user_query)

        return jsonify({
            "query": user_query,
            "response": answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3️⃣ Run Flask
#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port
        return s.getsockname()[1]  # Return the free port number

if __name__ == "__main__":
    port = find_free_port()  # Find an available port
    print(f"Running on port {port}")  # Print the port for reference
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

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
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

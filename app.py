from flask import Flask, render_template, request, jsonify
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set up Hugging Face API token
hf_api_token = os.getenv("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")
if not hf_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not set in environment variables")

# Initialize the Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=hf_api_token
)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful AI assistant. Answer the user's question as accurately and concisely as possible.\n\nUser: {user_input}\n\nAssistant:"
)

# Create chain
chain = prompt | llm | StrOutputParser()

# Store conversation history
conversation_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Get AI response
    try:
        response = chain.invoke({"user_input": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        return jsonify({"response": response, "history": conversation_history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


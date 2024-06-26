import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)  # Esto permite solicitudes CORS

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
parser = StrOutputParser()

template = """
    Answer the question based on the text below. If you can't answer the question, reply "I dont know".

    Context: {context}
    Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
embeddings = OpenAIEmbeddings()

index_name = "conver-index"
pinecone = PineconeVectorStore.from_existing_index(
    index_name=index_name, 
    embedding=embeddings, 
    text_key="conversation",
)
retrieval = pinecone.as_retriever()

setup = RunnableParallel(context=retrieval, question=RunnablePassthrough())
chain = setup | prompt | model | parser

@app.route('/query', methods=['POST'])
def query_pinecone():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        result = chain.invoke(question)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
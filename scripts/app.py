from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from flask_cors import CORS

# Set OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Load documents from specified directory
loader = PyPDFDirectoryLoader(path='path/to/your/pdf/directory')
data = loader.load()

# Initialize text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Create a vector store from the split documents using OpenAI embeddings
vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings())

# Initialize ChatOpenAI model with specified parameters
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7,
                 max_tokens=500, verbose=True)

app = Flask(__name__)
CORS(app)

# Define a route for creating footnotes


@app.route('/create-footnotes', methods=['POST'])
def create_footnotes():
    # Get the query from the request
    query = request.json.get('query')
    # Get the template from the request
    template = request.json.get('template')

    # Return error if template is missing
    if not template:
        return jsonify({"error": "Template parameter is missing!"}), 400

    # Build prompt using the provided template
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template)

    # Run the RetrievalQA chain with the provided prompt
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectorstore.as_retriever(),
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": query})
    # Return the result as a JSON response
    return jsonify(result["result"])


if __name__ == '__main__':
    app.run(debug=True)

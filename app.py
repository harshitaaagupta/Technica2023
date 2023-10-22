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

os.environ["OPENAI_API_KEY"] = ""

# Document loader
loader = PyPDFDirectoryLoader(path='../10-K')
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Store
vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings())

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7,
                 max_tokens=500, verbose=True)

app = Flask(__name__)
CORS(app)

@app.route('/create-footnotes', methods=['POST'])
def create_draft():
    query = request.json.get('query')
    template = request.json.get('template')

    if not template:
        return jsonify({"error": "Template parameter is missing!"}), 400

    # Build prompt using the provided template
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template)

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectorstore.as_retriever(),
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": query})
    return jsonify(result["result"])


if __name__ == '__main__':
    app.run(debug=True)
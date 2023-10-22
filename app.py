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
from google.cloud import storage 
import json 

#os.environ["OPENAI_API_KEY"] = ""
key_path = 'C:\\dev\\Projects\\APIKeys\\technica-cloud-key.json'

storage_client = storage.Client.from_service_account_json(key_path)

bucket_name = 'applecloud'
file_name = 'apple.jsonl'

# Split
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=4000, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# # Store
# vectorstore = Chroma.from_documents(
#     documents=all_splits, embedding=OpenAIEmbeddings())

# # Initialize the ChatOpenAI model
# llm = ChatOpenAI(model_name="gpt-4", temperature=0.7,
#                  max_tokens=500, verbose=True)

app = Flask(__name__)
CORS(app)

def check_bucket_file_access(bucket_name, file_name):
    try:
        # Create a client to access Google Cloud Storage
        client = storage.Client()

        # Access the specified bucket
        bucket = client.get_bucket(bucket_name)

        # Try to access the file
        blob = bucket.blob(file_name)
        blob.download_as_text()  # You can use download_to_filename() if it's a binary file

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

@app.route('/')
def check_access():
    if check_bucket_file_access(bucket_name, file_name):
        return "OK"
    else:
        return "Access to file failed."

if __name__ == '__main__':
    app.debug = True
    app.run()
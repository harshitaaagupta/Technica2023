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
key_path = 'C:\dev\Projects\API Keys'
storage_client = storage.Client.from_service_account_json(key_path)

bucket_name = 'apple-file'
file_name = 'apple.jsonl'

#Load the jsonl content from Google Cloud Storage
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(file_name)
json_content = blob.download_as_string()
data = json.loads(json_content)


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

@app.route('/')
def test_bucket_connection():
    try:
        # Create a client using the key file
        storage_client = storage.Client.from_service_account_json(key_path)

        # Get the bucket
        bucket = storage_client.get_bucket(bucket_name)

        # Get the blob (file)
        blob = bucket.blob(file_name)

        # Download the JSON content as a string
        json_content = blob.download_as_string()

        # Parse the JSON data
        data = json.loads(json_content)

        # If we reach this point, the connection and data retrieval were successful
        return "OK"

    except Exception as e:
        # If there was an error, print the error message
        return f"Error: {str(e)}

if __name__ == '__main__':
    app.run(debug=True)
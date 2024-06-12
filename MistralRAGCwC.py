#-----
#Install Required Packages
#-----
!pip install mistralai faiss-cpu python-docx flask


#-----
#Import Necessary Libraries
#-----
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import faiss
from docx import Document
import logging
from flask import Flask, request, jsonify


#-----
#Setup Logging
#-----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#-----
#Load and Preprocess the Text Data from Books
#-----
def load_docx(filepath):
    try:
        doc = Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return ""

book1_text = load_docx('/mnt/data/Coping with Crisis - Book 1 - Manual for Leaders (Reflow).docx')
book2_text = load_docx('/mnt/data/Coping with CRISIS - Book 2 - Manual for ICs (Reflow).docx')
book3_text = load_docx('/mnt/data/Coping with CRISIS - Book 3 - Workbook for Leaders (PDF).docx')
book4_text = load_docx('/mnt/data/Coping with CRISIS - Book 4 - Workbook for ICs (PDF).docx')

all_text = book1_text + "\n" + book2_text + "\n" + book3_text + "\n" + book4_text

chunk_size = 2048
chunks = [all_text[i:i + chunk_size] for i in range(0, len(all_text), chunk_size)]

#-----
#Create Embeddings for Text Chunks
#-----
client = MistralClient(api_key="YOUR_API_KEY")

def get_text_embedding(input_text):
    try:
        embeddings_batch_response = client.embeddings(
            model="mistral-embed",
            input=[input_text]
        )
        return embeddings_batch_response.data[0].embedding
    except Exception as e:
        logging.error(f"Error getting embedding for input: {input_text[:30]}...: {e}")
        return np.zeros(768)  # Assuming the embedding size is 768

text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])


#-----
#Store Embeddings in a Vector Database
#-----
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

#-----
#Create Embeddings for User Queries
#-----
def query_embeddings(question):
    try:
        return np.array([get_text_embedding(question)])
    except Exception as e:
        logging.error(f"Error getting embedding for query: {question}: {e}")
        return np.zeros((1, 768))  # Assuming the embedding size is 768

#-----
#Retrieve Relevant Chunks Based on Query
#-----
def retrieve_chunks(question_embedding, k=2):
    try:
        D, I = index.search(question_embedding, k)
        return [chunks[i] for i in I[0]]
    except Exception as e:
        logging.error(f"Error retrieving chunks: {e}")
        return []

#-----
#Generate Responses Using Retrieved Chunks
#-----
def generate_response(retrieved_chunks, question):
    prompt = f"""
    Context information is below.
    ---------------------
    {' '.join(retrieved_chunks)}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    try:
        messages = [ChatMessage(role="user", content=prompt)]
        chat_response = client.chat(model="mistral-medium-latest", messages=messages)
        return chat_response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "An error occurred while generating the response."

#-----
#Implement a User Interface for Querying
#-----
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question', '')
    question_embedding = query_embeddings(question)
    retrieved_chunks = retrieve_chunks(question_embedding)
    response = generate_response(retrieved_chunks, question)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

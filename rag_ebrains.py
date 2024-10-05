import fitz  # PyMuPDF
import nltk
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BartTokenizer, BartForConditionalGeneration
import streamlit as st


# Load the PDF
pdf_path = 'C://Users//ey290678//Ketli//Uni//MASTERARBEIT//Code//sodapdf-converted.pdf'
pdf_document = fitz.open(pdf_path)

# Extract text from each page
pdf_text = []
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    pdf_text.append(page.get_text())

# Close the PDF document
pdf_document.close()

# Join all the extracted text into a single string or split into paragraphs/sentences
pdf_text = " ".join(pdf_text)

#print(pdf_text)

nltk.download('punkt_tab')

# Function to chunk the text into manageable pieces
def chunk_text(text, max_chunk_size=512, overlap=50):
    sentences = nltk.sent_tokenize(text)  # Split into sentences
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        # Check if adding this sentence exceeds the max chunk size
        if current_chunk_size + sentence_length > max_chunk_size:
            # Append the current chunk to chunks and reset
            chunks.append(" ".join(current_chunk))
            # Start a new chunk with overlap
            current_chunk = current_chunk[-overlap:]  # Keep the last few sentences for context
            current_chunk_size = sum(len(s.split()) for s in current_chunk)

        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_chunk_size += sentence_length

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# Split the text into sentences
# documents = nltk.sent_tokenize(pdf_text)

# Split the text into chunks
documents = chunk_text(pdf_text, max_chunk_size=512, overlap=50)

# Alternatively, you can split it into paragraphs
# documents = pdf_text.split('\n\n')  # Split by double newlines (for paragraphs)

# Load a pre-trained embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create document embeddings
document_embeddings = embedding_model.encode(documents)

# Convert the embeddings to a format suitable for FAISS
embedding_dim = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Add the document embeddings to the index
faiss_index.add(np.array(document_embeddings))

def retrieve(query, k=5):
    # Encode the query into a vector
    query_embedding = embedding_model.encode([query])

    # Search FAISS index for the most relevant documents
    distances, indices = faiss_index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

model_name = 'facebook/bart-large-cnn'  # You can choose any BART model
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

model.eval()

def generate_response(query):
    # Retrieve relevant documents
    retrieved_docs = retrieve(query, k=2)

    # Combine the query with retrieved documents
    input_text = query + " ".join(retrieved_docs)

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024)

    # Generate the response
    outputs = model.generate(inputs['input_ids'], max_length=200, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Test the model
query = "what is virtual brain"
#response = generate_response(query)
#print("Response:", response)

### Generated response
# Response: what is virtual brainfind and share brain data comput model and softwar. map of the brain to navig and analys complex neuroscientif data. 
# find and share and work with medic and clinic brain data in a fulli compliant way.    “Virtual Brain’s” aim is to enabl breakthrough in differ
# area of brain scienc.

# Create centered main title 
st.title('Ask me a question 👩‍💻')

# Chat message storage
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message['content'])

prompt = st.chat_input("Input your prompt here")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    response=generate_response(prompt)
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})

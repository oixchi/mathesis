import fitz  # PyMuPDF
import nltk
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the PDF
pdf_path = 'C://Users//ey290678//Ketli//Uni//MASTERARBEIT//Code//Discover_EBRAINS_2023.pdf'
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

# Split the text into sentences
documents = nltk.sent_tokenize(pdf_text)

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

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(query):
    # Retrieve relevant documents
    retrieved_docs = retrieve(query, k=2)

    # Combine the query with retrieved documents
    input_text = query + " ".join(retrieved_docs)
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True)

    # Generate the response
    outputs = model.generate(inputs['input_ids'], max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Test the model
query = "Brain disorders are increasingly recognised as"
response = generate_response(query)
print("Response:", response)


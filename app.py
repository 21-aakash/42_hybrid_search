import streamlit as st
import os
import nltk
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Download necessary NLTK resources
nltk.download('punkt')

# Set up Pinecone API key using os.getenv()
api_key = os.getenv("PINECONE_API_KEY")

# Check if the API key is retrieved correctly
if not api_key:
    st.error("API key not found. Please set the PINECONE_API_KEY environment variable.")
    st.stop()

# Pinecone index name
index_name = "hybrid-search-langchain-pinecone"

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

# Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# BM25 Encoder
bm25_encoder = BM25Encoder().default()

# Sample sentences
sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited New Orleans",
]

# Fit the encoder on sentences
bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")

# Load the encoder values
bm25_encoder = BM25Encoder().load("bm25_values.json")

# Initialize Pinecone Hybrid Search Retriever
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index
)

# Add texts to the retriever
retriever.add_texts(sentences)

# Streamlit input
query = st.text_input("Enter your query:", "What city did I visit first?")

if query:
    result = retriever.invoke(query)
    st.write(f"Result: {result}")

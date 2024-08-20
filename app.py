import streamlit as st
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
import os
import nltk

nltk.download('punkt_tab')
# Initialize the Pinecone client

index_name = "hybrid-search-langchain-pinecone"
pc = Pinecone(api_key=api_key)

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Set up embeddings and BM25 encoder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

# Define sample sentences
sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited New Orleans",
]

# Fit the BM25 encoder
bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")

# Load the BM25 encoder from the saved file
bm25_encoder = BM25Encoder().load("bm25_values.json")

# Set up the retriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Add texts to the index
retriever.add_texts(sentences)

# Streamlit app layout
st.title("Hybrid Search with Langchain and Pinecone")

query = st.text_input("Enter your query:")
if st.button("Search"):
    results = retriever.invoke(query)
    if results:
        st.write("Top Search Result:", results[0].page_content)
    else:
        st.write("No results found.")

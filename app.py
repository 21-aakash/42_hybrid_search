import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import nltk
# Load environment variables
load_dotenv()
nltk.download('punkt_tab')
# Custom CSS for title font color
st.markdown(
    """
    <style>
    .title-font {
        color: #39FF14; /* Neon green color */
        font-size: 40px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit interface with custom title font color
st.markdown('<p class="title-font">🌐 Pinecone Hybrid Search with Streamlit</p>', unsafe_allow_html=True)
st.subheader("Explore hybrid search across dense and sparse embeddings")

# API keys and index details
api_key = os.getenv("PINECONE_API_KEY")
hf_token = os.getenv("HF_TOKEN")

index_name = "hybrid-search-langchain-pinecone"

# Initialize Pinecone client
try:
    with st.spinner("Initializing Pinecone client..."):
        pc = Pinecone(api_key=api_key)
except Exception as e:
    st.error(f"Error initializing Pinecone client: {e}")
    st.stop()

# Create the index if it doesn't exist
try:
    if index_name not in pc.list_indexes().names():
        with st.spinner("Creating Pinecone index..."):
            pc.create_index(
                name=index_name,
                dimension=384,  # dimensionality of dense model
                metric="dotproduct",  # sparse values supported only for dotproduct
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"Error creating or accessing the Pinecone index: {e}")
    st.stop()

# Vector embedding and sparse matrix
os.environ["HF_TOKEN"] = hf_token

# Initialize embeddings
with st.spinner("Initializing embeddings..."):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize BM25 encoder
with st.spinner("Initializing BM25 encoder..."):
    bm25_encoder = BM25Encoder().default()

# Example sentences
sentences = [
    "In 2023, I visited Delhi",
    "In 2022, I visited Mumbai",
    "In 2021, I visited Pune",
]

# Fit BM25 encoder on sentences
with st.spinner("Fitting BM25 encoder..."):
    bm25_encoder.fit(sentences)

# Store values to a JSON file
with st.spinner("Saving BM25 encoder values..."):
    bm25_encoder.dump("bm25_values.json")

# Load BM25 encoder
with st.spinner("Loading BM25 encoder values..."):
    bm25_encoder = BM25Encoder().load("bm25_values.json")

# Initialize retriever
with st.spinner("Initializing retriever..."):
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Add texts to the retriever
with st.spinner("Adding texts to the retriever..."):
    retriever.add_texts(sentences)

# Streamlit input for query
query = st.text_input("🔍 Enter your query:", "What city did I visit first?")

# Run the query and display the result
if st.button("Search"):
    with st.spinner("Searching..."):
        results = retriever.invoke(query)

    if results:
        st.success("Result Found!")
        st.markdown("### **Results:**")
        
        # Initialize a flag to check if a specific match is found
        specific_match_found = False
        
        for res in results:
            # Display only the sentence containing '2023' when queried
            if "2023" in query and "2023" in res.page_content:
                st.markdown(f"**Answer:** {res.page_content}")
                specific_match_found = True
                
            elif "2021" in query and "2021" in res.page_content:
                st.markdown(f"**Answer:** {res.page_content}")
                specific_match_found = True
        
        # If no specific match found, show all results
        if not specific_match_found:
            for i, res in enumerate(results, 1):
                st.markdown(f"**{i}.** {res.page_content}")
                st.markdown("---")
    else:
        st.warning("No results found for your query.")

# Add a footer
st.markdown(
    """
    ---
    **Pinecone Hybrid Search**
    """
)

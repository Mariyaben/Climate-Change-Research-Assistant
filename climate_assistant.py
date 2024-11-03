import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample dataset
document_texts = [
    "This study examines the accelerated melting of polar ice caps due to rising global temperatures, highlighting the potential impact on sea-level rise and global ecosystems.",
    "An overview of current carbon emission levels, renewable energy technologies, and policies aimed at reducing carbon footprints to mitigate climate change.",
    "Analyzes the adverse effects of global warming on marine biodiversity, including coral bleaching and species migration patterns.",
    "Explores the relationship between deforestation and climate change, focusing on greenhouse gas emissions and ecosystem disruption.",
    "Discusses the challenges and opportunities of renewable energy adoption in densely populated urban areas."
]
# Precompute document embeddings
doc_embeddings = model.encode(document_texts, convert_to_tensor=True)

# Function for document retrieval
def climate_research_assistant(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, doc_embeddings, top_k=5)
    results = [document_texts[hit['corpus_id']] for hit in hits[0]]
    return results

# Streamlit UI
st.title("Climate Change Research Assistant 🌍")
st.write("Enter a climate-related query to get quick insights on the topic.")

# User query input
user_query = st.text_input("Enter your query here:")

# Display results when user enters a query
if user_query:
    with st.spinner("Fetching insights..."):
        insights = climate_research_assistant(user_query)
        st.success("Here are the most relevant insights:")

        for idx, insight in enumerate(insights, 1):
            st.write(f"**Insight {idx}:** {insight}")

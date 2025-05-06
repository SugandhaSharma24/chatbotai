import streamlit as st
import PyPDF2
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sentence_transformers import SentenceTransformer

# ----- Load Models -----
@st.cache_resource
def load_model():
    return SentenceTransformer('distiluse-base-multilingual-cased-v1')  # Multilingual model

@st.cache_resource
def load_nlp():
    return spacy.load("xx_ent_wiki_sm")  # Multilingual spacy model that works with Arabic and English
@st.cache_resource
def load_melvis_model():
    # Load the MELVIS model and tokenizer
    model_name = "Geotrend/MELVIS-large"  # Example MELVIS model, you can choose others
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
# ----- Text Extraction -----
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# ----- Sentence Splitting -----
def split_text_into_sentences(text):
    # Basic sentence splitting using a period followed by space
    sentences = text.split(". ")  # Split based on period and space
    return sentences

# ----- Entity Extraction -----
def extract_entities(text_chunk, nlp):
    doc = nlp(text_chunk)
    return [(ent.text.strip(), ent.label_) for ent in doc.ents]

# ----- Knowledge Graph -----
def build_knowledge_graph(sentences, nlp):
    G = nx.Graph()
    for i, sentence in enumerate(sentences):
        ents = extract_entities(sentence, nlp)
        sentence_node = f"sentence_{i}"
        G.add_node(sentence_node, type="sentence")
        for ent_text, label in ents:
            if ent_text:
                G.add_node(ent_text, type=label)
                G.add_edge(sentence_node, ent_text)
    return G

def find_sentences_by_entity(query, graph, sentences):
    matches = [node for node in graph.nodes if query.lower() in node.lower()]
    related_sentences = set()
    for node in matches:
        for neighbor in graph.neighbors(node):
            if neighbor.startswith("sentence_"):
                idx = int(neighbor.replace("sentence_", ""))
                related_sentences.add(sentences[idx])
    return list(related_sentences)

# ----- Semantic Search -----
def create_embeddings(sentences, model):
    return model.encode(sentences)

def retrieve_relevant_sentences(query, model, index, sentences, k=2):  # Limit results to top 2
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [sentences[i] for i in I[0]]

# ----- Visualization -----
def plot_graph(graph):
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=0.8)
    node_colors = ['lightblue' if graph.nodes[n].get('type') == 'sentence' else 'lightgreen' for n in graph.nodes()]
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray', ax=ax)
    return fig

# ----- Streamlit App -----
st.set_page_config(page_title="PDF Knowledge Graph Chatbot", layout="wide")
st.title("🤖 PDF Knowledge Graph Chatbot")

uploaded_file = st.file_uploader("Upload your PDF (English or Arabic)", type="pdf")

if uploaded_file:
    # Step 1: Extract text from PDF and split it into sentences
    text = extract_text_from_pdf(uploaded_file)
    sentences = split_text_into_sentences(text)

    model = load_model()
    nlp = load_nlp()

    # Step 2: Build Knowledge Graph
    graph = build_knowledge_graph(sentences, nlp)

    # Step 3: Embed sentences for semantic search
    sentence_embeddings = create_embeddings(sentences, model)

    # Step 4: Create Faiss index for fast retrieval
    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(np.array(sentence_embeddings))

    st.success("✅ PDF processed and knowledge graph built.")

    with st.expander("📊 Show Knowledge Graph"):
        fig = plot_graph(graph)
        st.pyplot(fig)

    st.subheader("💬 Ask a question")
    query = st.text_input("Enter a topic or entity (in English or Arabic, e.g., 'age', 'العمر'):")

    if query:
        # Step 5: Search for relevant sentences using the query
        # Optionally, filter sentences by entities (can be improved for more complex queries)
        entity_filtered_sentences = find_sentences_by_entity(query, graph, sentences)

        # Step 6: Perform semantic search (use the filtered sentences or all sentences)
        relevant_sentences = retrieve_relevant_sentences(query, model, index, entity_filtered_sentences or sentences, k=2)

        if relevant_sentences:
            st.write("### 📌 Relevant Excerpts:")
            for i, sentence in enumerate(relevant_sentences):
                st.markdown(f"**Result {i+1}:** {sentence}")
        else:
            st.warning("No relevant data found for your query.")

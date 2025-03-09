import weaviate
from weaviate import Client
from sentence_transformers import SentenceTransformer
import torch
import os
from openai import OpenAI
import textwrap
from dotenv import load_dotenv
import re
import streamlit as st
import uuid
import json
import time
from pyngrok import ngrok
# Load environment variables
load_dotenv()

# ================ CONFIGURATION ================
# Weaviate connection parameters
WEAVIATE_URL = "https://v6romuf8rgsqk2qukrt0ga.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "aWRgXXZ1Is1MMPAADYcvoGlGrQjcflXnZXfQ"

# Model used for embeddings
MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'

# LLM configuration (using OpenAI's API)
OPENAI_API_KEY = "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"
LLM_MODEL = "gpt-4o"

# Weaviate class configuration
CLASS_NAME = "LegalDocument"

# Global model instance
embedding_model = None


# Global conversation history
@st.cache_resource
def get_conversation_history():
    return {}


def initialize_model():
    """Initialize the embedding model once."""
    global embedding_model

    with st.spinner("Loading embedding model..."):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load model with device specification
        embedding_model = SentenceTransformer(MODEL_NAME, device=device)
        return embedding_model


def connect_to_weaviate():
    """Connect to Weaviate database."""
    client_kwargs = {"url": WEAVIATE_URL}
    if WEAVIATE_API_KEY:
        client_kwargs["auth_client_secret"] = weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)

    client = Client(**client_kwargs)
    return client


def connect_to_llm():
    """Connect to the LLM API."""
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set it in your environment variables.")
        raise ValueError("OpenAI API key not found.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    return client


def create_query_embedding(query_text, model=None):
    """Create embedding for search query using the provided or global model instance."""
    global embedding_model

    if model is None and embedding_model is None:
        embedding_model = initialize_model()

    if model is None:
        model = embedding_model

    # Format query for e5 model
    formatted_query = f"query: {query_text}"

    # Generate embedding and normalize
    query_embedding = model.encode(formatted_query, normalize_embeddings=True)
    return query_embedding.tolist()


def detect_language(text):
    """Detect the language of the text (Arabic, French, or English)."""
    # Simple language detection based on character sets
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    french_chars = sum(1 for c in text if c in 'éèêëàâçùûüÿôœæ')

    if arabic_chars > len(text) * 0.4:
        return "arabic"
    elif french_chars > 0:
        return "french"
    else:
        return "english"


def hybrid_search(client, query, alpha=0.75, limit=8):
    """Perform hybrid search in Weaviate."""
    # Generate query embedding
    query_vector = create_query_embedding(query)
    vector_str = ", ".join(str(x) for x in query_vector)

    # Perform hybrid search using GraphQL
    graphql_query = {
        "query": f"""
        {{
          Get {{
            {CLASS_NAME}(
              hybrid: {{
                query: "{query}"
                alpha: {alpha}
                vector: [{vector_str}]
              }}
              limit: {limit}
            ) {{
              _additional {{
                id
                distance
              }}
              content
              originalChunkId
              documentId
              documentTitle
              article
            }}
          }}
        }}
        """
    }

    response = client.query.raw(graphql_query["query"])

    # Process results
    if response and "data" in response and "Get" in response["data"] and CLASS_NAME in response["data"]["Get"]:
        results = response["data"]["Get"][CLASS_NAME]

        # Format results for context
        contexts = []
        for result in results:
            context = {
                "content": result.get("content", ""),
                "document_id": result.get("documentId", ""),
                "document_title": result.get("documentTitle", ""),
                "article": result.get("article", ""),
                "chunk_id": result.get("originalChunkId", ""),
                "distance": result["_additional"]["distance"]
            }
            contexts.append(context)

        return contexts
    else:
        return []


def format_context_for_llm(contexts):
    """Format the retrieved contexts into a prompt for the LLM."""
    formatted_contexts = []

    for i, ctx in enumerate(contexts):
        # Format each context with its metadata
        context_text = f"""
DOCUMENT {i + 1}:
Title: {ctx['document_title']}
Article: {ctx['article']}
Content: {ctx['content']}
Reference: [Document ID: {ctx['document_id']}, Article: {ctx['article']}]
"""
        formatted_contexts.append(context_text)

    return "\n".join(formatted_contexts)


def extract_references(response_text, contexts):
    """Extract references from the response text and create a formatted references section."""
    # Extract references like [مدونة التجارة، المادة XXX]
    reference_pattern = r'\[(.*?)(المادة|Article)\s*(\d+)(.*?)\]'
    found_refs = re.findall(reference_pattern, response_text)

    # Create a set of unique article numbers
    unique_refs = set()
    for ref in found_refs:
        article_num = ref[2]  # The article number is the third group
        # Add tuple of doc title and article number
        doc_title = ref[0].strip().rstrip('،').strip()
        if not doc_title:
            doc_title = "مدونة التجارة"  # Default title if none found
        unique_refs.add((doc_title, article_num))

    # Create a references section
    references_text = ""

    # Match references with actual content from contexts
    for doc_title, article_num in sorted(unique_refs, key=lambda x: int(x[1])):
        # Find the content for this article
        article_content = ""
        for ctx in contexts:
            if ctx['article'] == article_num:
                article_content = ctx['content']
                break

        # Add to references text
        references_text += f"- {doc_title} (Article {article_num}): {article_content[:150]}{'...' if len(article_content) > 150 else ''}\n"

    return references_text


def generate_legal_response(llm_client, query, contexts, session_id=None, language="arabic"):
    """Generate a response from the LLM based on the query and retrieved contexts."""

    # Format the contexts
    formatted_contexts = format_context_for_llm(contexts)

    # Get conversation history for this session
    history = []
    conversation_history = get_conversation_history()

    if session_id and session_id in conversation_history:
        history = conversation_history[session_id][-5:]  # Get last 5 exchanges to maintain context

    # Create a system prompt specific to legal assistance
    system_prompt = f"""You are a specialized legal assistant focusing on Moroccan commercial law.
Your task is to answer legal questions accurately using ONLY the provided legal document contexts.

Follow these guidelines:
1. Base your answers ONLY on the provided legal documents
2. Cite specific articles and references using the format [مدونة التجارة، المادة X]
3. If the context doesn't contain relevant information, say so honestly
4. Be specific and precise - legal advice requires accuracy
5. Respond in {language} (the same language as the query)
6. Always mention when information might be incomplete
7. DO NOT include a separate references section - just cite in-line as [مدونة التجارة، المادة X]"""

    # Create the messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Add conversation history if available
    for msg in history:
        messages.append(msg)

    # Add the current query and context
    messages.append({"role": "user", "content": f"""
Question: {query}

Here are the relevant legal documents to consider when answering:

{formatted_contexts}

Please answer the question based only on these legal documents, with proper citations.
"""})

    # Generate the completion
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,  # Lower temperature for more factual responses
        max_tokens=1000
    )

    response_text = response.choices[0].message.content

    # Extract references
    references = extract_references(response_text, contexts)

    # Store the exchange in conversation history
    if session_id:
        conversation_history = get_conversation_history()
        if session_id not in conversation_history:
            conversation_history[session_id] = []

        conversation_history[session_id].append({"role": "user", "content": query})
        conversation_history[session_id].append({"role": "assistant", "content": response_text})

    return {
        "answer": response_text,
        "references": references
    }


def legal_assistant(query, session_id=None):
    """Main function that runs the entire legal assistant pipeline."""
    # Initialize connections
    weaviate_client = connect_to_weaviate()
    llm_client = connect_to_llm()

    # Detect language
    language = detect_language(query)

    # Retrieve relevant contexts using hybrid search
    contexts = hybrid_search(weaviate_client, query, alpha=0.75)

    if not contexts:
        return {
            "answer": "لم أتمكن من العثور على معلومات قانونية ذات صلة للإجابة على سؤالك."
            if language == "arabic" else
            "Je n'ai pas pu trouver d'informations juridiques pertinentes pour répondre à votre question."
            if language == "french" else
            "I couldn't find any relevant legal information to answer your question.",
            "references": ""
        }

    # Generate response using LLM
    response = generate_legal_response(llm_client, query, contexts, session_id, language)

    return response


def get_session_id():
    """Get or create a session ID for the user."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


# ... (keep all previous imports and functions the same)

def main():
    """Main Streamlit application with enhanced UI."""
    # Page configuration
    st.set_page_config(
        page_title=" Moroccan Legal AI Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Session ID
    session_id = get_session_id()

    # Initialize model only once
    if "model_initialized" not in st.session_state:
        with st.spinner("🔄 Initializing AI systems..."):
            initialize_model()
            st.session_state.model_initialized = True

    # ---- Professional CSS Styling ----
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Modern Professional Theme */
    * {{
        font-family: 'Inter', sans-serif;
    }}

    /* Main Container */
    .main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }}

    /* Chat Messages */
    .user-message {{
        background: #1A237E;
        color: white;
        border-radius: 15px 15px 0 15px;
        margin: 1rem 0 1rem auto;
        max-width: 75%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}

    .assistant-message {{
        background: #F8F9FA;
        color: #2D3436;
        border-radius: 15px 15px 15px 0;
        margin: 1rem auto 1rem 0;
        max-width: 75%;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}

    /* RTL Support */
    .rtl {{
        direction: rtl;
        text-align: right;
        font-family: 'Amiri', serif;
    }}

    /* References Section */
    .references {{
        background: #FFFFFF;
        border-left: 4px solid #1A237E;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 8px;
        font-size: 0.9em;
    }}

    /* Input Area */
    .stTextInput>div>div>input {{
        border-radius: 25px !important;
        padding: 1rem 2rem !important;
        font-size: 1.1em !important;
        border: 2px solid #E0E0E0 !important;
    }}

    /* Language Selector */
    .lang-btn {{
        border: none !important;
        background: #E8EAF6 !important;
        color: #1A237E !important;
        border-radius: 20px !important;
        padding: 0.5rem 1.5rem !important;
        margin: 0 0.5rem !important;
        transition: all 0.3s ease !important;
    }}

    .lang-btn.active {{
        background: #1A237E !important;
        color: white !important;
    }}

    /* Progress Spinner */
    .stSpinner>div>div {{
        border-color: #1A237E !important;
    }}

    /* Header */
    .header {{
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1A237E, #303F9F);
        color: white;
        margin-bottom: 2rem;
        border-radius: 0 0 30px 30px;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ---- App Header ----
    st.markdown("""
    <div class="header">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚖️ Moroccan Legal AI Assistant</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">Advanced AI-powered Legal Research & Analysis System</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Language Selector ----
    cols = st.columns(5)
    with cols[1]:
        en_btn = st.button("🇬🇧 English", key="en_btn",
                           help="Switch to English interface")
    with cols[2]:
        fr_btn = st.button("🇫🇷 Français", key="fr_btn",
                           help="Passer à l'interface française")
    with cols[3]:
        ar_btn = st.button("🇲🇦 العربية", key="ar_btn",
                           help="التبديل إلى الواجهة العربية")

    # Set default language
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if en_btn: st.session_state.lang = "en"
    if fr_btn: st.session_state.lang = "fr"
    if ar_btn: st.session_state.lang = "ar"

    # ---- Chat History ----
    chat_container = st.container()
    with chat_container:
        # Display welcome message
        if not st.session_state.get("messages"):
            welcome_msg = {
                "en": "Welcome to the Moroccan Legal AI Assistant. How may I assist you with commercial law matters today?",
                "fr": "Bienvenue à l'Assistant Juridique IA Marocain. Comment puis-je vous aider avec des questions de droit commercial aujourd'hui?",
                "ar": "مرحبًا بكم في المساعد القانوني المغربي. كيف يمكنني مساعدتك في قضايا القانون التجاري اليوم؟"
            }
            st.session_state.messages = [{
                "role": "assistant",
                "content": welcome_msg[st.session_state.lang]
            }]

        # Display messages
        for msg in st.session_state.messages:
            direction = "rtl" if detect_language(msg["content"]) == "arabic" else ""
            with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else "👤"):
                st.markdown(f'<div class="{direction}">{msg["content"]}</div>',
                            unsafe_allow_html=True)
                if "references" in msg and msg["references"]:
                    with st.expander("🔍 View Legal References", expanded=False):
                        st.markdown(msg["references"])

    # ---- User Input ----
    with st.container():
        placeholder = {
            "en": "Ask your legal question...",
            "fr": "Posez votre question juridique...",
            "ar": "اطرح سؤالك القانوني..."
        }[st.session_state.lang]

        if query := st.chat_input(placeholder, key="user_input"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})

            # Process query
            with st.spinner("🔍 Analyzing legal documents..."):
                try:
                    response = legal_assistant(query, session_id)
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "references": response["references"]
                    })
                except Exception as e:
                    st.error(f"⚠️ System Error: {str(e)}")

            st.rerun()

    # ---- Footer ----
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: #666; font-size: 0.9rem;">
        <hr style="margin-bottom: 1rem;">
        <div>© 2025 Moroccan Legal AI Assistant</div>
        <div style="margin-top: 0.5rem;">
            <span style="margin: 0 0.5rem;">AI-Powered Legal Research</span> • 
            <span style="margin: 0 0.5rem;">v2.1.0</span> • 
            <span style="margin: 0 0.5rem;">Not Legal Advice</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
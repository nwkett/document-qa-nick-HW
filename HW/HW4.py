import streamlit as st
from openai import OpenAI, AuthenticationError
import sys
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def extract_text_from_html(html_path):
    "Extract text from a HTML file"
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Clean up text
            text = " ".join(text.split())
            return text
    except Exception as e:
        st.error(f"Error reading {html_path}: {str(e)}")
        return None

def add_to_collection(collection, text, file_name):
    try:

        client = st.session_state.openai_client
        

        response = client.embeddings.create(
            input=text,
            model='text-embedding-3-small'
        )
        

        embedding = response.data[0].embedding
        
        collection.add(
            documents=[text],
            ids=[file_name],
            embeddings=[embedding]
        )
        return True
    except Exception as e:
        st.error(f"Error adding {file_name} to collection: {str(e)}")
        return False


def load_htmls_to_collection(folder_path, collection):
    """Load all htmls from folder into the collection"""
    # Check if collection is empty and load PDFs
    if collection.count() == 0:
        html_files = list(Path(folder_path).glob("*.html"))
        
        if not html_files:
            st.warning(f"No HTML files found in {folder_path}")
            return False
        
        all_chunks = []
        for html_path in html_files:
            # Extract text from HTML
            text = extract_text_from_html(html_path)
            
            if text:
                # Add to collection
                chunks= chunk_text(text, html_path.name, num_chunks = 4)
                all_chunks.extend(chunks)
        
        if all_chunks:
            fixed_size_chunks_to_collection(collection, all_chunks)
            st.success(f"✅ Successfully loaded {len(html_files)} HTML files ({len(all_chunks)} chunks) into ChromaDB")
        
            return True
        else:
            st.info(f"Collection already contains {collection.count()} documents")
            return True
        
def chunk_text(text, file_name, num_chunks=2):
    """I used fixed-size chunking for this assignment because the directions
    asked for 2 mini documents for each. So splitting them into 2 chunks will be easiest since
    the documents are simple and only a few sections long. Additionally, this is a simple
    chunking method.

    The tradeoff is that it may split the document in the middle of a sentence or section
    but since the document is about student orgs, theyre smaller and structured the same way across the board.
    """


    chunk_size = len(text) // num_chunks
    chunks = []
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(text)
        chunk = text[start:end].strip()
        
        if chunk:  
            chunks.append({
                'text': chunk,
                'id': f"{file_name}_chunk_{i+1}"
            })
    
    return chunks

def fixed_size_chunks_to_collection(collection, chunks):
    "Add chunks to collection"

    try:
        client = st.session_state.openai_client

        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]

        response = client.embeddings.create(
            input = texts,
            model = 'text-embedding-3-small'
        )

        embeddings = [item.embedding for item in response.data]

        collection.add(
            documents=texts,
            ids=ids,
            embeddings=embeddings
        )
        return True
    except Exception as e:
        st.error(f"Error adding chunks to collection:  {str(e)}")
        return False


if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if 'HW4_VectorDB' not in st.session_state:
    with st.spinner("Initializing ChromaDB and loading PDFs..."):
        chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
        collection = chroma_client.get_or_create_collection('HW4Collection')
    

        load_htmls_to_collection('./HW4-Data/', collection)
        
        st.session_state.Lab4_VectorDB = collection


def apply_buffer():
    MAX_HISTORY = 10
    "Changed the max history to 10 so the last 5 interactions are involved"  

    msgs = st.session_state.messages
    system_msg = msgs[:1]
    rest = msgs[1:]    

    if len(rest) > MAX_HISTORY:
        rest = rest[-MAX_HISTORY:]

    st.session_state.messages = system_msg + rest



st.title('Nicks Lab 4')

# Sidebar
openAI_model = st.sidebar.selectbox("Select Model", ('mini', 'regular'))

if openAI_model == 'mini':
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = 'gpt-4o'



# Lab 3 chat bot 

SYSTEM_PROMPT = """
are a helpful assistant for Syracuse University's student organizations.
When answering questions:
1. If you use information from the provided context, clearly state: "Based on the student organization information..."
2. If answering from general knowledge, state: "I don't have specific information about this, but..."
3. Be concise, simple and helpful
"""

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Hi! I'm your course information assistant. Ask me anything about the Syracuse iSchool courses!"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("Ask about course topics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    client = st.session_state.openai_client
    collection = st.session_state.Lab4_VectorDB
    
    response = client.embeddings.create(
        input=prompt,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  
    )
    
    context = ""
    if results['documents'] and len(results['documents'][0]) > 0:
        context = "\n\n---\n\n".join(results['documents'][0])
        sources = results['ids'][0]
        
        context_message = f"""
        Use the following context from course materials to answer the question. If the answer is in this context, make sure to say "Based on the course materials..." 
        
        Context:
        {context}
        
        Sources: {', '.join(sources)}
        """
        
        st.session_state.messages.insert(-1, {"role": "system", "content": context_message})
    
    apply_buffer()
    
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=st.session_state.messages,
        stream=True
    )
    
    with st.chat_message("assistant"):
        response_text = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
  
    st.session_state.messages = [
        msg for msg in st.session_state.messages 
        if not (msg["role"] == "system" and "Context:" in msg["content"])
    ]
    
    apply_buffer()
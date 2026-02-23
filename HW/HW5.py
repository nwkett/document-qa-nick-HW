import streamlit as st
from openai import OpenAI, AuthenticationError
import sys
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#PDF Functions
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # Clean up text (remove extra whitespace)
        text = " ".join(text.split())
        return text
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {str(e)}")
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


def load_pdfs_to_collection(folder_path, collection):
    """Load all PDFs from folder into the collection"""
    # Check if collection is empty and load PDFs
    if collection.count() == 0:
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        
        if not pdf_files:
            st.warning(f"No PDF files found in {folder_path}")
            return False
        
        for pdf_path in pdf_files:
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            
            if text:
                # Add to collection
                add_to_collection(collection, text, pdf_path.name)
        
        st.success(f"✅ Successfully loaded {len(pdf_files)} PDF files into ChromaDB")
        return True
    else:
        st.info(f"Collection already contains {collection.count()} documents")
        return True
    

# Initialize AI Client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if 'HW5_VectorDB' not in st.session_state:
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
    collection = chroma_client.get_or_create_collection('HW5Collection')
    load_pdfs_to_collection('./HW-05-Data/', collection)
    st.session_state.HW5_VectorDB = collection

# Step 3 Vector Search
def relevant_course_info(query):
    client = st.session_state.openai_client
    collection = st.session_state.HW5_VectorDB

    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    if results['documents'] and len(results['documents'][0]) > 0:
        context = "\n\n---\n\n".join(results['documents'][0])
        sources = ", ".join(results['ids'][0])
        return f"Sources: {sources}\n\n{context}"
    else:
        return "No relevant course materials found."


    

# Tool function for LLM about retrieving info from courses
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_course_info",
            "description": "Retrieves relevant course material from the Syracuse iSchool course documents based on a query. Use this whenever the user asks about course content or course-related topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the course materials"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# Short Term Memory Buffer
def apply_buffer():
    MAX_HISTORY = 4  

    msgs = st.session_state.messages
    system_msg = msgs[:1]
    rest = msgs[1:]    

    if len(rest) > MAX_HISTORY:
        rest = rest[-MAX_HISTORY:]

    st.session_state.messages = system_msg + rest


st.title('Nicks HW5')

# Sidebar
openAI_model = st.sidebar.selectbox("Select Model", ('mini', 'regular'))

if openAI_model == 'GPT4':
    model_to_use = "gpt-4o"
else:
    model_to_use = 'gpt-4o-mini'



# Lab 3 chat bot 

SYSTEM_PROMPT = """
You are a helpful course information assistant for Syracuse University's School of Information Studies that utilizes the loaded PDFs.

When answering questions:
1. If you use information from the course materials provided in the context, clearly state: "Based on this course's course materials..."
2. If you're answering from general knowledge (not from the provided PDFs), clearly state: "Based on my general knowledge..." or "I don't have specific information about this in the course materials, but..."
3. Be concise and helpful
4. If you're unsure or don't have information in the provided materials, say so clearly
"""

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Hi! I'm your course information assistant. Ask me anything about the Syracuse iSchool courses!"}
    ]

# Chat History

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    if msg["role"] == "tool":
        continue
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])


# Chat Input

if prompt := st.chat_input("Ask about course topics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    client = st.session_state.openai_client
    apply_buffer()
    

    response = client.chat.completions.create(
        model=model_to_use,
        messages=st.session_state.messages,
        tools=tools
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        query_arg = tool_call.function.arguments["query"]

        result = relevant_course_info(query_arg)

        st.session_state.messages.append(message)
        st.session_state.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })


        final_response = client.chat.completions.create(
                model=model_to_use,
                messages=st.session_state.messages,
                stream=True
            )

        with st.chat_message("assistant"):
            response_text = st.write_stream(final_response)

    else:
        response_text = message.content
        with st.chat_message("assistant"):
            st.markdown(response_text)
   
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    apply_buffer()
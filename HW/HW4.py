import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI, AuthenticationError
import google.generativeai as genai


st.title('Nicks Lab3 Question answering chatbot')

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def apply_buffer():
    MAX_HISTORY = 6  
    msgs = st.session_state.messages
    system_msg = msgs[:1]   
    rest = msgs[1:]         

    if len(rest) > MAX_HISTORY:
        rest = rest[-MAX_HISTORY:]

    st.session_state.messages = system_msg + rest

def build_system_prompt_with_urls(url_text: str) -> str:
    if url_text and url_text.strip():
        return BASE_SYSTEM_PROMPT + "\n\nURL CONTEXT (from the user's URLs):\n" + url_text.strip()
    return BASE_SYSTEM_PROMPT


def call_openai(messages):
    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages,
        stream=True
    )
    return stream

def call_gemini(messages):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_to_use)

    system_text = messages[0]["content"]
    convo_lines = [f"SYSTEM: {system_text}"]

    for m in messages[1:]:
        role = m["role"].upper()
        convo_lines.append(f"{role}: {m['content']}")

    prompt_text = "\n".join(convo_lines)

    return model.generate_content(prompt_text, stream=True)

BASE_SYSTEM_PROMPT = """
You are a helpful chatbot.

Rules:
1) Explain everything so a 10-year-old can understand. Use simple words and short sentences.
2) After you answer a question, ALWAYS ask: "Do you want more info?"
3) If the user says "Yes", provide more information about the SAME topic, then ask again: "Do you want more info?"
4) If the user says "No", say: "Okay! What can I help you with?" and wait for a new question.

If URL context is provided, use it as your main source of truth.
If the URL context does not contain the answer, say you could not find it in the URLs.
""".strip()

st.sidebar.header("Options")

# URLs
st.sidebar.subheader("Input up to two URLs")
url1 = st.sidebar.text_input("URL 1", placeholder="https://...")
url2 = st.sidebar.text_input("URL 2", placeholder="https://...")
load_urls = st.sidebar.button("Load URL(s)")

st.sidebar.subheader("Pick an LLM")
vendor = st.sidebar.selectbox("LLMs", ("OpenAI", "Gemini"))

if vendor == "OpenAI":
    model_to_use = st.sidebar.selectbox(
        "OpenAI premium model",
        ("gpt-5-chat-latest",)  
    )
else:
    model_to_use = st.sidebar.selectbox(
        "Gemini premium model",
        ("gemini-3-pro-preview",)      
    )

if "url_text" not in st.session_state:
    st.session_state.url_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": build_system_prompt_with_urls(st.session_state.url_text)},
        {"role": "assistant", "content": "Hi! What can I help you with?"}
    ]

if "expecting_more_info" not in st.session_state:
    st.session_state.expecting_more_info = False

if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""


if load_urls:
    texts = []

    if url1.strip():
        t1 = read_url_content(url1.strip())
        if t1:
            texts.append(f"URL 1 ({url1.strip()}):\n{t1}")

    if url2.strip():
        t2 = read_url_content(url2.strip())
        if t2:
            texts.append(f"URL 2 ({url2.strip()}):\n{t2}")

    st.session_state.url_text = "\n\n---\n\n".join(texts)

    st.session_state.messages[0]["content"] = build_system_prompt_with_urls(st.session_state.url_text)

    if st.session_state.url_text.strip():
        st.sidebar.success("Loaded URL text and updated system context!")
    else:
        st.sidebar.warning("No URL text loaded (check your URLs).")

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


YES_SET = {"yes", "y", "yeah", "yep", "sure", "ok", "okay"}
NO_SET = {"no", "n", "nope", "nah"}

if prompt := st.chat_input("Type here..."):
    user_text = prompt.strip()
    normalized = user_text.lower().strip()

    if st.session_state.expecting_more_info and normalized in YES_SET:
        user_text = (
            f"Give more information about this topic: {st.session_state.last_topic}. "
            "Explain for a 10-year-old. End with: Do you want more info?"
        )

    elif st.session_state.expecting_more_info and normalized in NO_SET:
        st.session_state.expecting_more_info = False
        st.session_state.last_topic = ""

        assistant_text = "Okay! What can I help you with?"
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

        apply_buffer()

        with st.chat_message("assistant"):
            st.write(assistant_text)

        st.stop()

    else:
        st.session_state.last_topic = user_text
        st.session_state.expecting_more_info = True

    st.session_state.messages.append({"role": "user", "content": user_text})

    apply_buffer()

    with st.chat_message("assistant"):
        if vendor == "OpenAI":
            stream = call_openai(st.session_state.messages)
            response = st.write_stream(stream)
        else:
            gstream = call_gemini(st.session_state.messages)
            chunks = []
            for chunk in gstream:
                if hasattr(chunk, "text") and chunk.text:
                    chunks.append(chunk.text)
                    st.write(chunk.text)
            response = "".join(chunks).strip()

    st.session_state.messages.append({"role": "assistant", "content": response})
    apply_buffer()
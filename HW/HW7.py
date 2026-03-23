import streamlit as st
import pandas as pd
import chromadb
import anthropic


@st.cache_data
def load_articles():
    return pd.read_csv("HW/news.csv").fillna("")

@st.cache_resource
def load_vector_db(_df):
    collection = chromadb.Client().create_collection("news")
    for i, row in _df.iterrows():
        doc = str(row["Document"])
        title, description = doc.split("Description:", 1) if "Description:" in doc else (doc, "")
        collection.add(
            documents=[f"{title.strip()}. {description.strip()}"],
            metadatas=[{"title": title.strip(), "company": str(row["company_name"]), "date": str(row["Date"]), "url": str(row["URL"])}],
            ids=[str(i)]
        )
    return collection

def build_prompt(question, collection):
    results = collection.query(query_texts=[question], n_results=5)
    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"Company: {meta['company']}\nTitle: {meta['title']}\nDate: {meta['date']}\nContent: {doc[:600]}\nURL: {meta['url']}\n\n"

    return f"""You are a bot that will be used by a large, global law firm to monitor news about its clients. Use ONLY the articles below to answer.

{context}
Question: {question}

If asked for interesting news, rank by interest and importance. Always cite the article"""

# Main 
st.title("HW 7 News Bot")

model = st.sidebar.radio("Model", ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"])

df = load_articles()
collection = load_vector_db(df)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about the news...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    response = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"]).messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": build_prompt(user_input, collection)}]
    )
    answer = response.content[0].text

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})
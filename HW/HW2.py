import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI, AuthenticationError
import google.generativeai as genai


# Validate Open AI key function for lab, used below
def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        #random openAI function to check validity
        return True, "API key is valid"
    except AuthenticationError:
        return False, "Invalid API key"
    except Exception as e:
        return False, f"Error validating API key: {str(e)}"
    

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def google_gen(model_type, question_to_ask):
    # Get API key
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing google_apikey in Streamlit secrets.")
        st.stop()

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Select model based on type
    if "pro" in model_type:
        model = genai.GenerativeModel("gemini-3-pro-preview")
    elif "lite" in model_type:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
    else:
        model = genai.GenerativeModel("gemini-3-flash-preview")


    # Gemini expects a single prompt
    response = model.generate_content(f"URL content:\n\n{page_text}\n\n---\n\n{effective_question_language}")

    return response.text

# Side bar controls

summary_selection = st.sidebar.radio(
    "Choose summary format:",
    [
        "100-word summary",
        "Two connected paragraphs",
        "Five bullet points",
    ],
)

llm_choice = st.sidebar.selectbox(
    "Choose LLM:",
    ["OpenAI", "Gemini"]  
)

output_language = st.sidebar.selectbox(
    "Output language:",
    ["English", "French", "Spanish"]
)

use_advanced_model = st.sidebar.checkbox("Use advanced model")

# Show title and description.
st.title("Nick's HW 2")
st.write(
    "Enter a URL"
)

url = st.text_input("Enter a webpage URL:", placeholder="https://example.com")


openai_api_key = st.secrets.get("OPENAI_API_KEY")
gemini_api_key = st.secrets.get("GEMINI_API_KEY")

if llm_choice == "OpenAI":
    if not openai_api_key:
        st.error("Missing OPENAI_API_KEY. Add it to Streamlit secrets.")
        st.stop()
    is_valid, message = validate_api_key(openai_api_key)
    if is_valid:
        st.success(message, icon="✅")
    else:
        st.error(message)
        st.stop()
elif llm_choice == "Gemini":
    if not gemini_api_key:
        st.error("Missing GEMINI_API_KEY. Add it to Streamlit secrets.")
        st.stop()
    st.success("Gemini key found in secrets.", icon="✅")

# if valid, continue

if st.button("Generate response", disabled=not url):
    page_text = read_url_content(url)

    if not page_text:
        st.stop()
    
    language = f"Write the summary in {output_language}"
    
    if summary_selection == "100-word summary":
        effective_question = (
            "Summarize the document in 100 words. "
        )

    elif summary_selection == "Two connected paragraphs":
        effective_question = (
            "Summarize the document in two connected paragraphs."
        )

    else:
        effective_question = (
            "Summarize the document in exactly five bullet points. "
            "Each bullet should be one key idea."
        )
 
    effective_question_language = f"{effective_question} {language}"

    if llm_choice == "OpenAI":
        # Create an OpenAI client.
        client = OpenAI(api_key=openai_api_key)
        model_name = "gpt-4o" if use_advanced_model else "gpt-4o-mini"
        
        messages = [
            {
                "role": "user",
                "content": f"URL content:\n\n{page_text}\n\n---\n\n{effective_question_language}"
            }
        ]
        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)

    elif llm_choice == "Gemini":
        model_name = "pro" if use_advanced_model else "lite"
        response = google_gen(model_name, effective_question_language)
        st.write(response)


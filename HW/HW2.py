import streamlit as st
import fitz
from openai import OpenAI, AuthenticationError
from io import BytesIO

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
    
def extract_text_from_pdf(pdf_file):
    document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ''
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Side bar controls

summary_selection = st.sidebar.radio(
    "Choose summary format:",
    [
        "100-word summary",
        "Two connected paragraphs",
        "Five bullet points",
    ],
)

use_advanced_model = st.sidebar.checkbox("Use advanced model (4o)")

# Show title and description.
st.title("Nick's document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)


openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit secrets.")
    st.stop()

# Validate API key when entered
if openai_api_key:
    is_valid, message = validate_api_key(openai_api_key)
    if is_valid:
        st.success(message, icon="✅")
    else:
        st.error(message)
        st.stop()
else:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# if valid, continue
if openai_api_key:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)



    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf .txt or .md)", type=("txt", "md", "pdf")
    )


    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file:

        # Process the uploaded file and question.
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document_text = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document_text = extract_text_from_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

    model_name = "gpt-4o" if use_advanced_model else "gpt-4o-mini"

    if summary_selection == "100-word summary":
        effective_question = (
            "Summarize the document in 100 words. "
        )

    elif summary_selection == "Two connected paragraphs":
        effective_question = (
            "Summarize the document in two connected paragraphs."
        )

    elif summary_selection == "Five bullet points":
        effective_question = (
            "Summarize the document in exactly five bullet points. "
            "Each bullet should capture one key idea."
        )

    else:
        effective_question = question


        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document_text} \n\n---\n\n {effective_question}",
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

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

# Show title and description.
st.title("Nick's document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")

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

    if uploaded_file and question:

        # Process the uploaded file and question.
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = extract_text_from_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)

import streamlit as st

st.set_page_config(page_title="HW Manager", layout="wide")

# Create a navigation object with pages
pg = st.navigation(
    [
        st.Page("HW/HW1.py", title="HW 1", icon="1️⃣"),
        st.Page("HW/HW2.py", title="HW 2", icon="2️⃣", default=True),
    ]
)

# Run the selected page
pg.run()
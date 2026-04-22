import streamlit as st

st.set_page_config(page_title="HW Manager", layout="wide")

pg = st.navigation(
    [
        st.Page("HW/HW1.py", title="HW 1", icon="1️⃣"),
        st.Page("HW/HW2.py", title="HW 2", icon="2️⃣"),
        st.Page("HW/HW3.py", title="HW 3", icon="2️⃣"),
        st.Page("HW/HW4.py", title="HW 4", icon="2️⃣"),
        st.Page("HW/HW5.py", title="HW 5", icon="2️⃣"),
        st.Page("HW/HW7.py", title="HW 7", icon="2️⃣"),
        st.Page("HW/FINAL.py", title="Final Project", icon="2️⃣", default=True),

    ]
)

pg.run()
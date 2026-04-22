
import io
import os
from dataclasses import dataclass
 
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
 
load_dotenv()
 
# ─────────────────────────────────────────────────
# Contacts
# ─────────────────────────────────────────────────
 
@dataclass
class Contact:
    company_name: str
    primary_contact: str
    email: str
    description: str
 
    @property
    def greeting_name(self) -> str:
        return self.primary_contact if self.primary_contact.strip() else "Team"
 
 
def parse_contacts(csv_text: str) -> list[Contact]:
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
 
    required = {"company_name", "primary_contact", "email", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
 
    return [
        Contact(
            company_name=str(r["company_name"]).strip(),
            primary_contact=str(r["primary_contact"]).strip()
                if pd.notna(r["primary_contact"]) else "",
            email=str(r["email"]).strip(),
            description=str(r["description"]).strip(),
        )
        for _, r in df.iterrows()
    ]
 
 
# ─────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────
 
ddg_search = DuckDuckGoSearchRun(
    name="company_research",
    description=(
        "Search the web for recent news, social media posts, blog articles, "
        "or any public info about a company. Use this to find a specific, "
        "personalized opener for the email — a recent event they hosted, "
        "an Instagram post, a blog, a news mention, or a milestone. "
        "Pass the company name plus keywords like 'recent event', 'news', "
        "'Instagram', or 'blog'."
    ),
)
 
TOOLS = [ddg_search]
 
 
# ─────────────────────────────────────────────────
# System prompts — PVC framework
# ─────────────────────────────────────────────────
 
OUTREACH_PROMPT = """\
You are writing a cold outreach email on behalf of Summit Standard, \
an event rentals company.
 
FRAMEWORK — PVC (Personalization, Value, Call-to-Action):
 
1. PERSONALIZATION (opener): Write a specific, non-generic first line that \
proves you actually know who they are. Use the company_research tool to find \
a trigger event, recent news, Instagram post, website content, or blog post. \
NEVER use a generic opener like "I came across your company" or "I love what \
you do." If the search returns nothing useful, reference something concrete \
from their company description instead.
 
2. VALUE (one sentence): "Summit Standard supports event planners with \
high-quality products and expert-level operational excellence."
 
3. CALL-TO-ACTION (low friction): "Are you available for a 15-minute chat \
next week?"
 
RULES:
- Greet with "Hi {greeting_name}," — if greeting name is "Team", use \
"Dear [Company] Team,".
- UNDER 100 WORDS in the body. Hard limit. Be crisp.
- Do NOT list packages or pricing. This is a first touch, not a catalog.
- Sign off: [Your Name], Summit Standard
- PROCESS: ALWAYS call company_research first to personalize the opener. \
Then draft the email.
- Output ONLY the final email:
 
Subject: <subject line>
 
<email body>
"""
 
FOLLOWUP_PROMPT = """\
You are writing a follow-up email on behalf of Summit Standard, \
an event rentals company. The recipient did not respond to a prior outreach.
 
FRAMEWORK — PVC (Personalization, Value, Call-to-Action):
 
1. PERSONALIZATION: Reference the previous email briefly ("I reached out \
recently about supporting your upcoming events…") then add ONE new specific \
detail you found via the company_research tool — a recent post, event, hire, \
or news item. Show you are still paying attention to their world.
 
2. VALUE (one sentence): "We help event teams like yours deliver seamless \
experiences with premium rentals and hands-on support."
 
3. CALL-TO-ACTION: "Would a 15-minute call this week or next work for you?"
 
RULES:
- Greet with "Hi {greeting_name},"
- UNDER 75 WORDS. This is a nudge, not a re-pitch.
- Confident, not apologetic. Never write "just checking in", "sorry to \
bother", "bumping this", or "circling back".
- Sign off: [Your Name], Summit Standard
- PROCESS: Call company_research to find something fresh to reference.
- Output ONLY the final email:
 
Subject: <subject line>
 
<email body>
"""
 
 
# ─────────────────────────────────────────────────
# LLM + Agent
# ─────────────────────────────────────────────────
 
def get_model():
    provider = st.session_state.get("provider", "OpenAI")
    api_key = st.session_state.get("api_key", "")
 
    if provider == "OpenAI":
        return ChatOpenAI(
            model=st.session_state.get("model", "gpt-4o-mini"),
            temperature=0.6,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=st.session_state.get("model", "claude-sonnet-4-20250514"),
            temperature=0.6,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )
 
 
def draft_email(contact: Contact, mode: str) -> str:
    system_prompt = FOLLOWUP_PROMPT if mode == "Follow-Up" else OUTREACH_PROMPT
    model = get_model()
 
    agent = create_agent(
        model,
        tools=TOOLS,
        system_prompt=system_prompt,
    )
 
    user_message = (
        f"Draft an email for this contact.\n\n"
        f"Company name: {contact.company_name}\n"
        f"Greeting name: {contact.greeting_name}\n"
        f"Email: {contact.email}\n"
        f"Company description: {contact.description}\n\n"
        f"Start by researching {contact.company_name} with the "
        f"company_research tool, then draft the email using the PVC framework."
    )
 
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )
 
    # Get the final AI message from the response
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return "Error: agent did not produce a response."
 
 
# ─────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────
 
st.set_page_config(
    page_title="EventReach — Summit Standard",
    page_icon="🏔️",
    layout="centered",
)
 
st.markdown("""
<style>
.block-container { max-width: 720px; }
div[data-testid="stChatMessage"] { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)
 
st.title("🏔️ EventReach")
st.caption("Summit Standard — personalized outreach powered by AI research.")
 
# ── Sidebar ──────────────────────────────────────
 
with st.sidebar:
    st.header("Settings")
 
    provider = st.radio("LLM", ["OpenAI", "Anthropic"], horizontal=True)
    st.session_state["provider"] = provider
 
    if provider == "OpenAI":
        st.session_state["api_key"] = st.text_input(
            "API key", type="password", placeholder="sk-...",
            value=os.getenv("OPENAI_API_KEY", ""),
        )
        st.session_state["model"] = st.selectbox(
            "Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        )
    else:
        st.session_state["api_key"] = st.text_input(
            "API key", type="password", placeholder="sk-ant-...",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
        )
        st.session_state["model"] = st.selectbox(
            "Model", ["claude-sonnet-4-20250514", "claude-haiku-4-20250414"],
        )
 
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Upload your contacts CSV\n"
        "2. Pick a contact from the dropdown\n"
        "3. Choose Outreach or Follow-Up\n"
        "4. The agent searches DuckDuckGo for their company, "
        "then drafts a PVC email using what it finds"
    )
 
# ── CSV upload ───────────────────────────────────
 
uploaded = st.file_uploader("Upload contacts CSV", type=["csv"])
 
if not uploaded:
    st.info(
        "Upload a CSV with columns: "
        "**Company_Name**, **Primary_Contact**, **Email**, **Description**"
    )
    st.stop()
 
try:
    contacts = parse_contacts(uploaded.getvalue().decode("utf-8"))
except Exception as e:
    st.error(f"CSV error: {e}")
    st.stop()
 
# ── Contact picker + mode ────────────────────────
 
col1, col2 = st.columns([3, 2])
 
with col1:
    contact_map = {}
    display_names = []
    for c in contacts:
        label = f"{c.company_name} — {c.greeting_name}"
        contact_map[label] = c
        display_names.append(label)
 
    selected_label = st.selectbox("Contact", display_names)
    selected = contact_map[selected_label]
 
with col2:
    mode = st.selectbox("Email type", ["Outreach", "Follow-Up"])
 
with st.expander(f"📋 {selected.company_name}", expanded=False):
    st.markdown(f"**Contact:** {selected.greeting_name}")
    st.markdown(f"**Email:** {selected.email}")
    st.markdown(f"**About:** {selected.description}")
 
# ── Chat history ─────────────────────────────────
 
if "messages" not in st.session_state:
    st.session_state["messages"] = []
 
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
# ── Draft button ─────────────────────────────────
 
if st.button(f"✍️ Draft {mode} → {selected.company_name}", type="primary"):
    if not st.session_state.get("api_key"):
        st.warning("Enter your API key in the sidebar.")
        st.stop()
 
    user_msg = (
        f"Draft a **{mode.lower()}** email for "
        f"**{selected.company_name}** (Dear {selected.greeting_name})."
    )
    st.session_state["messages"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)
 
    with st.chat_message("assistant"):
        with st.spinner(
            f"Researching {selected.company_name} and drafting {mode.lower()}…"
        ):
            try:
                draft = draft_email(selected, mode)
            except Exception as e:
                draft = f"⚠️ Error: {e}"
        st.markdown(draft)
 
    st.session_state["messages"].append({"role": "assistant", "content": draft})
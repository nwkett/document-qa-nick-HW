import io
from datetime import datetime

import chromadb
import pandas as pd
import streamlit as st
from langchain.agents import create_agent
from langchain_community.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun


# 
# Upload CSV
# 

def parse_contacts(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = df.columns.str.strip()

    required = {"Company_Name", "Primary_Contact", "Email", "Description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["Primary_Contact"] = df["Primary_Contact"].fillna("").astype(str).str.strip()

    return df


def greeting_name(row) -> str:
    name = row["Primary_Contact"]
    return name if name.strip() else "Team"



# ChromaDB


client = chromadb.PersistentClient(path="./email_memory")
collection = client.get_or_create_collection("sent_emails")


def save_email(company_name: str, email_type: str, draft: str):
    """Save a drafted email to ChromaDB for future reference."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc_id = f"{company_name}_{email_type}_{timestamp}".replace(" ", "_")

    collection.add(
        documents=[draft],
        metadatas=[{
            "company": company_name,
            "type": email_type,
            "date": timestamp,
        }],
        ids=[doc_id],
    )


def get_past_emails(company_name: str) -> list[dict]:
    """Retrieve all past emails sent to a company, sorted by date."""
    results = collection.get(
        where={"company": company_name},
        include=["documents", "metadatas"],
    )

    if not results["documents"]:
        return []

    emails = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        emails.append({
            "date": meta["date"],
            "type": meta["type"],
            "content": doc,
        })

    emails.sort(key=lambda x: x["date"])
    return emails


def rerank_emails(emails: list[dict], mode: str) -> list[dict]:
    """Rerank past emails by prioritizing the same type and most recent first."""
    same_type = [e for e in emails if e["type"] == mode]
    other_type = [e for e in emails if e["type"] != mode]

    same_type.sort(key=lambda x: x["date"], reverse=True)
    other_type.sort(key=lambda x: x["date"], reverse=True)

    return same_type + other_type


def format_past_emails(emails: list[dict]) -> str:
    """Format past emails into a string for the prompt."""
    if not emails:
        return "No previous emails have been sent to this company."

    parts = []
    for i, email in enumerate(emails, 1):
        parts.append(
            f"--- Email {i} ({email['type']}, sent {email['date']}) ---\n"
            f"{email['content']}"
        )
    return "\n\n".join(parts)



# Tools


def get_tools():
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
    return [ddg_search]



# System prompts PVC framework


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

You have access to the PREVIOUS EMAILS we sent to this company. Use them \
to write a follow-up that builds on what was already said — reference the \
specific subject, value prop, or CTA from the last email. Do NOT repeat \
the same pitch. Each follow-up should add something new.

FRAMEWORK — PVC (Personalization, Value, Call-to-Action):

1. PERSONALIZATION: Reference what you said in the last email specifically \
(the actual subject line or opener, not a vague "I reached out recently"). \
Then add ONE new detail from the company_research tool — a recent post, \
event, hire, or news item.

2. VALUE (one sentence): Restate briefly but differently from the last email. \
"We help event teams like yours deliver seamless experiences with premium \
rentals and hands-on support."

3. CALL-TO-ACTION: "Would a 15-minute call this week or next work for you?"

RULES:
- Greet with "Hi {greeting_name},"
- UNDER 75 WORDS. This is a nudge, not a re-pitch.
- Confident, not apologetic. Never write "just checking in", "sorry to \
bother", "bumping this", or "circling back".
- Do NOT repeat the same opener, value statement, or CTA verbatim from \
any previous email. Vary your approach each time.
- Sign off: [Your Name], Summit Standard
- PROCESS: Read the previous emails carefully, then call company_research \
to find something fresh, then draft.
- Output ONLY the final email:

Subject: <subject line>

<email body>
"""

ETHICS_PROMPT = """\
You are an ethics reviewer for outbound sales emails. Evaluate the following 
email draft against this rubric and return a brief pass/fail verdict with 
one sentence of reasoning for each criterion.

ETHICS RUBRIC:
1. Honesty — Does the email make any false or misleading claims?
2. Respect — Is the tone professional and non-manipulative?
3. Privacy — Does it reference any sensitive or inappropriate personal data?
4. Transparency — Is it clear this is a sales outreach from a real company?

Return your evaluation in this format:
1. Honesty: PASS/FAIL — reason
2. Respect: PASS/FAIL — reason
3. Privacy: PASS/FAIL — reason
4. Transparency: PASS/FAIL — reason

Overall: PASS/FAIL

EMAIL TO EVALUATE:
{draft}
"""

# LLM + Agent

def get_model():
    provider = st.session_state.get("provider", "OpenAI")
    api_key = st.session_state.get("api_key", "")

    if provider == "OpenAI":
        return ChatOpenAI(
            model=st.session_state.get("model", "gpt-4o-mini"),
            temperature=0.6,
            api_key=api_key or st.secrets.get("OPENAI_API_KEY", ""),
        )
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=st.session_state.get("model", "claude-sonnet-4-20250514"),
            temperature=0.6,
            api_key=api_key or st.secrets.get("ANTHROPIC_API_KEY", ""),
        )


def run_ethics_check(draft: str, model) -> str:
    """Run an ethics rubric evaluation on the drafted email."""
    prompt = ETHICS_PROMPT.format(draft=draft)
    response = model.invoke(prompt)
    return response.content


def draft_email(row, mode: str) -> tuple[str, str]:
    system_prompt = FOLLOWUP_PROMPT if mode == "Follow-Up" else OUTREACH_PROMPT
    model = get_model()
    greeting = greeting_name(row)
    company = row["Company_Name"]

    if "conversation_memory" not in st.session_state:
        st.session_state["conversation_memory"] = {}

    if company not in st.session_state["conversation_memory"]:
        st.session_state["conversation_memory"][company] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

    memory = st.session_state["conversation_memory"][company]

    agent = create_agent(
        model,
        tools=get_tools(),
        system_prompt=system_prompt,
    )

    past_emails = get_past_emails(company)
    reranked_emails = rerank_emails(past_emails, mode)
    history_text = format_past_emails(reranked_emails)

    buffer_history = memory.load_memory_variables({}).get("chat_history", [])
    buffer_text = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in buffer_history]
    ) if buffer_history else "No prior conversation."

    user_message = (
        f"Draft an email for this contact.\n\n"
        f"Company name: {company}\n"
        f"Greeting name: {greeting}\n"
        f"Email: {row['Email']}\n"
        f"Company description: {row['Description']}\n\n"
        f"PREVIOUS EMAILS SENT TO THIS COMPANY:\n"
        f"{history_text}\n\n"
        f"CONVERSATION HISTORY:\n"
        f"{buffer_text}\n\n"
        f"Start by researching {company} with the "
        f"company_research tool, then draft the email using the PVC framework."
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )

    messages = result.get("messages", [])
    draft = "Error: agent did not produce a response."
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            draft = msg.content
            break

    memory.save_context({"input": user_message}, {"output": draft})

    ethics_result = run_ethics_check(draft, model)

    save_email(company, mode, draft)

    return draft, ethics_result



# Streamlit UI


st.set_page_config(page_title="EventReach", layout="centered")

st.title("EventReach")

# Sidebar

with st.sidebar:
    st.header("Settings")

    provider = st.radio("LLM", ["OpenAI", "Anthropic"], horizontal=True)
    st.session_state["provider"] = provider

    if provider == "OpenAI":
        st.session_state["api_key"] = st.text_input(
            "API key", type="password", placeholder="sk-...",
            value=st.secrets.get("OPENAI_API_KEY", ""),
        )
        st.session_state["model"] = st.selectbox(
            "Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        )
    else:
        st.session_state["api_key"] = st.text_input(
            "API key", type="password", placeholder="sk-ant-...",
            value=st.secrets.get("ANTHROPIC_API_KEY", ""),
        )
        st.session_state["model"] = st.selectbox(
            "Model", ["claude-sonnet-4-20250514", "claude-haiku-4-20250414"],
        )

    st.divider()
    if st.button("Clear Conversation Memory"):
        st.session_state["conversation_memory"] = {}
        st.success("Conversation memory cleared.")

# CSV upload

uploaded = st.file_uploader("Upload contacts CSV", type=["csv"])

if not uploaded:
    st.info(
        "Upload a CSV with columns: "
        "Company_Name, Primary_Contact, Email, Description"
    )
    st.stop()

try:
    df = parse_contacts(uploaded.getvalue().decode("utf-8"))
except Exception as e:
    st.error(f"CSV error: {e}")
    st.stop()

# Contact picker + mode

col1, col2 = st.columns([3, 2])

with col1:
    labels = [
        f"{row['Company_Name']} — {greeting_name(row)}"
        for _, row in df.iterrows()
    ]
    selected_label = st.selectbox("Contact", labels)
    selected_idx = labels.index(selected_label)
    selected_row = df.iloc[selected_idx]

with col2:
    mode = st.selectbox("Email type", ["Outreach", "Follow-Up"])

# Show email history for selected company

past = get_past_emails(selected_row["Company_Name"])
if past:
    st.write(f"Email history ({len(past)} sent)")
    for email in past:
        st.text(f"[{email['type']}] {email['date']}")

# Chat history

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Draft button

if st.button(f"Draft {mode} for {selected_row['Company_Name']}"):
    if not st.session_state.get("api_key"):
        st.warning("Enter your API key in the sidebar.")
        st.stop()

    greeting = greeting_name(selected_row)
    user_msg = (
        f"Draft a {mode.lower()} email for "
        f"{selected_row['Company_Name']} (Dear {greeting})."
    )
    st.session_state["messages"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        try:
            draft, ethics_result = draft_email(selected_row, mode)
        except Exception as e:
            draft = f"Error: {e}"
            ethics_result = ""
        st.markdown(draft)

        if ethics_result:
            with st.expander("Ethics Rubric Evaluation"):
                st.markdown(ethics_result)

    st.session_state["messages"].append({"role": "assistant", "content": draft})
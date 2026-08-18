import streamlit as st

from chatbot import (
    process_message,
    clear_filters,
    reset_state,
    get_data_status,
    new_state,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TNEA Engineering College Assistant",
    page_icon="\U0001f393",
    layout="wide",
)


# ============================================================
# SESSION STATE INITIALISATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    # Sent to OpenRouter — bounded to last MAX_HISTORY turns inside chatbot.py
    st.session_state.chat_history = []

if "chat_state" not in st.session_state:
    st.session_state.chat_state = new_state()


state = st.session_state.chat_state


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("\U0001f3af Your Preferences")

    cutoff = state.get("cutoff")
    if cutoff is not None:
        st.metric("Cutoff", f"{cutoff:g}")
    else:
        st.write("**Cutoff:** Not set")

    category = state.get("category")
    st.write(f"**Category:** {category or 'Not set'}")

    st.write(f"**District:** {state.get('district') or 'Any'}")
    st.write(f"**Branch:** {state.get('branch') or 'Any'}")
    st.write(f"**College Type:** {state.get('college_type') or 'Any'}")
    st.write(f"**Results:** Top {state.get('limit', 10)}")

    selected = state.get("selected_college")
    if selected:
        st.write(f"**College Code:** {selected}")

    st.divider()

    if st.button("\U0001f9f9 Clear Filters", use_container_width=True):
        clear_filters(state)
        st.rerun()

    if st.button("\U0001f504 Reset Chat", use_container_width=True):
        reset_state(state)
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    with st.expander("\U0001f4ca Data Status"):
        st.write(get_data_status())


# ============================================================
# HEADER
# ============================================================

st.title("\U0001f393 TNEA Engineering College Assistant")
st.caption(
    "Find engineering colleges based on your TNEA cutoff, "
    "community, district, and branch."
)


# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# WELCOME MESSAGE  (shown only on fresh start)
# ============================================================

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            """\
**Welcome to the TNEA Engineering College Assistant!** \U0001f393

Tell me your **TNEA cutoff mark** to get started:

`my cutoff is 160`   or just   `160`

I'll guide you through **category \u2192 district \u2192 branch \u2192 results** step by step.

---

Or ask directly:

`160, BC, Chennai, CSE, top 10`

`college code 2347`

`what is TNEA?`

`government colleges in Coimbatore`
"""
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_message = st.chat_input("Ask about TNEA colleges...")


# ============================================================
# PROCESS MESSAGE
# ============================================================

if user_message:

    st.session_state.messages.append(
        {"role": "user", "content": user_message}
    )

    with st.spinner("Thinking..."):
        response, updated_history = process_message(
            user_message,
            state,
            st.session_state.chat_history,
        )

    st.session_state.chat_history = updated_history

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()
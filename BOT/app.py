"""
app.py  —  TNEA Saarthi  •  Premium AI College Counselling Assistant
─────────────────────────────────────────────────────────────────────
UI:      Dark glassmorphism · Purple/violet accent · Streamlit
Backend: chatbot.py (ZERO changes — all logic preserved as-is)
─────────────────────────────────────────────────────────────────────
"""

import streamlit as st

from chatbot import (
    process_message,
    clear_filters,
    reset_state,
    get_data_status,
    validate_data,
    new_state,
    ALL_RECORDS,
    COMMUNITY_DISPLAY,
)


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TNEA Saarthi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
# DATA STATS  (cached — never re-reads 15k records on each rerun)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _get_data_stats():
    total     = len(ALL_RECORDS)
    districts = len({r.get("district", "") for r in ALL_RECORDS if r.get("district")})
    branches  = len({r.get("branch", "")   for r in ALL_RECORDS if r.get("branch")})
    return total, districts, branches


_total_recs, _total_dists, _total_branches = _get_data_stats()
_data_ok = _total_recs > 0


# ═══════════════════════════════════════════════════════════════
# PREMIUM CSS — Injected once, governs the entire UI
# ═══════════════════════════════════════════════════════════════

_CSS = """
<style>
/* ── Google Font ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Variables ───────────────────────────────────────── */
:root {
  --bg:         #05050f;
  --bg2:        #0a0a1e;
  --card:       rgba(255,255,255,0.04);
  --card-hover: rgba(255,255,255,0.07);
  --purple:     #7c3aed;
  --purple-l:   #a78bfa;
  --purple-d:   #5b21b6;
  --blue:       #3b82f6;
  --green:      #10b981;
  --amber:      #f59e0b;
  --red:        #ef4444;
  --txt:        #e2e8f0;
  --txt2:       #94a3b8;
  --txt3:       #64748b;
  --border:     rgba(124,58,237,0.18);
  --border2:    rgba(255,255,255,0.06);
}

/* ── Global ──────────────────────────────────────────── */
html, body, .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  background: var(--bg) !important;
}

.stApp {
  background: linear-gradient(160deg, #05050f 0%, #07071a 60%, #05050f 100%) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.35); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.55); }

/* Selection */
::selection { background: rgba(124,58,237,0.35); color:#fff; }

/* ── Hide default Streamlit chrome ──────────────────── */
#MainMenu, footer, .stDeployButton { display:none !important; }
header[data-testid="stHeader"] { display:none !important; }

/* ── Main content area ───────────────────────────────── */
.main .block-container {
  padding: 1.5rem 2rem 6rem 2rem !important;
  max-width: 900px !important;
}

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0a20 0%, #07071a 100%) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div:first-child {
  padding-top: 0 !important;
}

/* ── Chat messages ───────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.3rem 0 !important;
  animation: fadeUp 0.25s ease;
}

[data-testid="stChatMessageContent"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(12px) !important;
  padding: 1rem 1.25rem !important;
  color: var(--txt) !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
}

/* Tables inside chat */
[data-testid="stChatMessageContent"] table {
  width: 100% !important;
  border-collapse: collapse !important;
  margin: 0.5rem 0 !important;
  font-size: 0.82rem !important;
}
[data-testid="stChatMessageContent"] th {
  background: rgba(124,58,237,0.25) !important;
  color: #e2e8f0 !important;
  padding: 0.55rem 0.7rem !important;
  border: 1px solid rgba(124,58,237,0.2) !important;
  font-size: 0.76rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
}
[data-testid="stChatMessageContent"] td {
  background: rgba(255,255,255,0.02) !important;
  color: #cbd5e1 !important;
  padding: 0.5rem 0.7rem !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stChatMessageContent"] tr:hover td {
  background: rgba(124,58,237,0.08) !important;
}

/* Headings inside chat */
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 {
  color: #f1f5f9 !important;
  margin-top: 0.75rem !important;
}

/* Blockquote (disclaimer) */
[data-testid="stChatMessageContent"] blockquote {
  border-left: 3px solid var(--purple) !important;
  background: rgba(124,58,237,0.07) !important;
  border-radius: 0 8px 8px 0 !important;
  padding: 0.5rem 0.85rem !important;
  margin: 0.75rem 0 0 !important;
  color: var(--txt2) !important;
  font-size: 0.82rem !important;
}

/* Code / inline code */
[data-testid="stChatMessageContent"] code {
  background: rgba(124,58,237,0.15) !important;
  color: var(--purple-l) !important;
  border-radius: 4px !important;
  padding: 0.15em 0.4em !important;
  font-size: 0.85em !important;
}

/* ── All Streamlit buttons → primary purple ──────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--purple) 0%, var(--purple-d) 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  font-size: 0.84rem !important;
  padding: 0.45rem 0.9rem !important;
  transition: all 0.2s ease !important;
  font-family: inherit !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #6d28d9 0%, #4c1d95 100%) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 5px 18px rgba(124,58,237,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Chat input ──────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(124,58,237,0.35) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: 0 0 20px rgba(124,58,237,0.08) !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] textarea {
  color: #e2e8f0 !important;
  background: transparent !important;
  font-size: 0.94rem !important;
  font-family: inherit !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #475569 !important;
}
/* Send button */
[data-testid="stChatInput"] button {
  background: var(--purple) !important;
  border-radius: 10px !important;
}

/* ── Metrics ─────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: rgba(124,58,237,0.08) !important;
  border: 1px solid rgba(124,58,237,0.2) !important;
  border-radius: 12px !important;
  padding: 0.65rem 0.75rem !important;
}
[data-testid="stMetricLabel"] p { color: var(--txt2) !important; font-size:0.75rem !important; }
[data-testid="stMetricValue"]   { color: var(--txt) !important;  font-size:1.1rem !important; font-weight:600 !important; }

/* ── Expanders ───────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary { color: var(--txt2) !important; }

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: rgba(124,58,237,0.15) !important; margin: 0.75rem 0 !important; }

/* ── Global text colours ─────────────────────────────── */
p, li, span, label { color: var(--txt2) !important; }
strong { color: var(--txt) !important; }
h1, h2, h3, h4 { color: #f1f5f9 !important; }

/* ── Spinner ─────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--purple) !important; }

/* ═══════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ═══════════════════════════════════════════════════════ */

/* Saarthi header bar */
.saarthi-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.9rem 1.5rem;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 16px;
  margin-bottom: 1.25rem;
  backdrop-filter: blur(12px);
}
.saarthi-header-left {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}
.saarthi-avatar-sm {
  width: 42px; height: 42px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--purple), var(--blue));
  display: flex; align-items: center; justify-content: center;
  font-size: 1.2rem;
  box-shadow: 0 0 14px rgba(124,58,237,0.5);
  flex-shrink: 0;
}
.saarthi-brand-name {
  font-size: 1.05rem;
  font-weight: 700;
  color: #f1f5f9 !important;
  margin: 0;
}
.saarthi-brand-tag {
  font-size: 0.72rem;
  color: var(--purple-l) !important;
  margin: 0;
}
.saarthi-header-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.data-pill {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.75rem;
  border-radius: 20px;
  border: 1px solid;
  font-size: 0.75rem;
  font-weight: 500;
}
.data-pill-green {
  background: rgba(16,185,129,0.1);
  border-color: rgba(16,185,129,0.3);
  color: #10b981 !important;
}
.data-pill-red {
  background: rgba(239,68,68,0.1);
  border-color: rgba(239,68,68,0.3);
  color: #ef4444 !important;
}
.dot-pulse {
  width: 7px; height: 7px;
  border-radius: 50%;
  display: inline-block;
}
.dot-green { background: #10b981; box-shadow: 0 0 6px rgba(16,185,129,0.7); }
.dot-red   { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.7); }

/* Sidebar branding block */
.sb-brand {
  text-align: center;
  padding: 1.5rem 1rem 1rem;
  border-bottom: 1px solid var(--border2);
  margin-bottom: 1rem;
}
.sb-avatar {
  width: 68px; height: 68px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--purple), var(--blue));
  display: flex; align-items: center; justify-content: center;
  font-size: 1.9rem;
  margin: 0 auto 0.75rem;
  box-shadow: 0 0 22px rgba(124,58,237,0.45);
}
.sb-name {
  font-size: 1.15rem;
  font-weight: 700;
  color: #f1f5f9 !important;
  margin: 0 0 0.2rem;
}
.sb-tagline {
  font-size: 0.72rem;
  color: var(--purple-l) !important;
  margin: 0;
}

/* Sidebar section label */
.sb-section {
  font-size: 0.68rem;
  font-weight: 700;
  color: var(--txt3) !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  padding: 0.5rem 0 0.3rem;
  margin-bottom: 0.25rem;
}

/* Profile rows */
.profile-card {
  background: var(--card);
  border: 1px solid var(--border2);
  border-radius: 12px;
  padding: 0.6rem 0.85rem;
  margin-bottom: 0.65rem;
}
.profile-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.3rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
.profile-row:last-child { border-bottom: none; }
.profile-lbl { font-size: 0.74rem; color: var(--txt3) !important; }
.profile-val { font-size: 0.82rem; color: #e2e8f0 !important; font-weight: 500; }
.profile-val-notset { font-size: 0.82rem; color: var(--txt3) !important; font-style: italic; }

/* Status badge */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.7rem;
  border-radius: 20px;
  font-size: 0.73rem;
  font-weight: 600;
  margin: 0.5rem 0 0.75rem;
}
.badge-ready {
  background: rgba(16,185,129,0.12);
  border: 1px solid rgba(16,185,129,0.28);
  color: #10b981 !important;
}
.badge-partial {
  background: rgba(245,158,11,0.12);
  border: 1px solid rgba(245,158,11,0.28);
  color: #f59e0b !important;
}

/* Data summary card */
.data-card {
  background: var(--card);
  border: 1px solid var(--border2);
  border-radius: 12px;
  padding: 0.65rem 0.85rem;
  margin-bottom: 0.65rem;
}
.data-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.25rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 0.78rem;
}
.data-row:last-child { border-bottom: none; }
.data-key { color: var(--txt3) !important; }
.data-val { color: #e2e8f0 !important; font-weight: 600; }

/* Nav buttons in sidebar */
.sb-nav-btn {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  width: 100%;
  padding: 0.5rem 0.75rem;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 9px;
  color: var(--txt2) !important;
  font-size: 0.83rem;
  cursor: pointer;
  transition: all 0.18s ease;
  margin-bottom: 0.15rem;
  font-family: inherit;
  text-align: left;
}
.sb-nav-btn:hover {
  background: rgba(124,58,237,0.12);
  border-color: rgba(124,58,237,0.25);
  color: #e2e8f0 !important;
  transform: translateX(3px);
}

/* Welcome card */
.welcome-card {
  background: rgba(124,58,237,0.06);
  border: 1px solid rgba(124,58,237,0.2);
  border-radius: 20px;
  padding: 2rem 2rem 1.5rem;
  text-align: center;
  animation: fadeUp 0.4s ease;
  margin-bottom: 1.5rem;
}
.welcome-avatar {
  width: 72px; height: 72px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--purple), var(--blue));
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem;
  margin: 0 auto 1rem;
  box-shadow: 0 0 30px rgba(124,58,237,0.5);
}
.welcome-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #f1f5f9 !important;
  margin: 0 0 0.3rem;
}
.welcome-sub {
  font-size: 0.88rem;
  color: var(--purple-l) !important;
  margin: 0 0 1.25rem;
}
.welcome-desc {
  font-size: 0.87rem;
  color: var(--txt2) !important;
  margin-bottom: 1.25rem;
  line-height: 1.6;
}
.welcome-chips {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 1.25rem;
}
.welcome-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.3rem 0.7rem;
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--border2);
  border-radius: 20px;
  font-size: 0.78rem;
  color: var(--txt2) !important;
}

/* Quick action cards (welcome screen) */
.qa-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
  margin-top: 0.5rem;
}
.qa-card {
  background: var(--card);
  border: 1px solid var(--border2);
  border-radius: 12px;
  padding: 0.85rem;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none;
}
.qa-card:hover {
  background: rgba(124,58,237,0.1);
  border-color: rgba(124,58,237,0.3);
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(124,58,237,0.15);
}
.qa-icon { font-size: 1.25rem; margin-bottom: 0.35rem; }
.qa-title { font-size: 0.82rem; font-weight: 600; color: #e2e8f0 !important; margin-bottom: 0.15rem; }
.qa-desc  { font-size: 0.72rem; color: var(--txt3) !important; }

/* Quick prompts strip (above input) */
.qp-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 0.75rem;
}
.qp-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.28rem 0.65rem;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(124,58,237,0.2);
  border-radius: 20px;
  font-size: 0.76rem;
  color: var(--txt2) !important;
  cursor: pointer;
  transition: all 0.18s ease;
  font-family: inherit;
  white-space: nowrap;
}
.qp-chip:hover {
  background: rgba(124,58,237,0.14);
  border-color: rgba(124,58,237,0.4);
  color: #e2e8f0 !important;
  transform: translateY(-1px);
}

/* Profile summary card (shown in chat after info collected) */
.psum {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}
.psum-item {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.25rem 0.6rem;
  background: rgba(124,58,237,0.1);
  border: 1px solid rgba(124,58,237,0.25);
  border-radius: 6px;
  font-size: 0.78rem;
  color: var(--purple-l) !important;
  font-weight: 500;
}

/* Footer */
.sb-footer {
  text-align: center;
  padding: 0.75rem 0;
  font-size: 0.7rem;
  color: var(--txt3) !important;
  border-top: 1px solid var(--border2);
  margin-top: 0.5rem;
}

/* ── Animations ───────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity:0; transform: translateY(10px); }
  to   { opacity:1; transform: translateY(0); }
}
@keyframes pulse {
  0%,100% { box-shadow: 0 0 6px rgba(16,185,129,0.5); }
  50%      { box-shadow: 0 0 12px rgba(16,185,129,0.9); }
}
.dot-green { animation: pulse 2.5s infinite; }

/* ── Mobile responsive ───────────────────────────────── */
@media (max-width: 768px) {
  .main .block-container { padding: 1rem 1rem 5rem !important; }
  .qa-grid { grid-template-columns: 1fr; }
  .saarthi-header { flex-direction: column; gap: 0.6rem; text-align: center; }
  [data-testid="stChatMessageContent"] { padding: 0.75rem 1rem !important; }
  table { display: block; overflow-x: auto; }
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════

if "messages"         not in st.session_state: st.session_state.messages         = []
if "chat_history"     not in st.session_state: st.session_state.chat_history     = []
if "chat_state"       not in st.session_state: st.session_state.chat_state       = new_state()
if "_pending_message" not in st.session_state: st.session_state._pending_message = None

state = st.session_state.chat_state


# ═══════════════════════════════════════════════════════════════
# HELPER: render HTML without touching Streamlit state
# ═══════════════════════════════════════════════════════════════

def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _pv(value, fallback: str = "Not set", fmt: str = "{}") -> str:
    """Format a profile value; return 'Not set' if None/empty."""
    if value is None or str(value).strip() in {"", "Any", "any"}:
        return fallback
    return fmt.format(value)


# ═══════════════════════════════════════════════════════════════
# PROCESS & DISPLAY  (unchanged logic — only wrapper changes)
# ═══════════════════════════════════════════════════════════════

def _process_and_display(user_message: str) -> None:
    """Appends user message → calls process_message → appends response → reruns."""
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.spinner("Saarthi is thinking..."):
        response, updated_history = process_message(
            user_message,
            state,
            st.session_state.chat_history,
        )
    st.session_state.chat_history = updated_history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Branding ────────────────────────────────────────────
    _html("""
    <div class="sb-brand">
      <div class="sb-avatar">🎓</div>
      <p class="sb-name">TNEA Saarthi</p>
      <p class="sb-tagline">Your AI College Counselling Companion</p>
    </div>
    """)

    # ── Profile section ──────────────────────────────────────
    cutoff  = state.get("cutoff")
    cat     = state.get("category")
    dist    = state.get("district")
    branch  = state.get("branch")
    ctype   = state.get("college_type")
    sel_col = state.get("selected_college")
    offset  = state.get("recommendation_offset", 0)

    cat_disp = COMMUNITY_DISPLAY.get(cat, cat) if cat else None

    is_ready = cutoff is not None and cat is not None
    badge_cls  = "badge-ready"   if is_ready else "badge-partial"
    badge_dot  = "🟢"            if is_ready else "🟡"
    badge_txt  = "Profile Ready" if is_ready else "Profile Incomplete"

    _html(f"""
    <p class="sb-section">MY PROFILE</p>
    <div class="profile-card">
      <div class="profile-row">
        <span class="profile-lbl">🎯 Cutoff</span>
        <span class="{'profile-val' if cutoff else 'profile-val-notset'}">{_pv(cutoff, "Not set", "{:g}")}</span>
      </div>
      <div class="profile-row">
        <span class="profile-lbl">👤 Category</span>
        <span class="{'profile-val' if cat else 'profile-val-notset'}">{_pv(cat_disp)}</span>
      </div>
      <div class="profile-row">
        <span class="profile-lbl">📍 District</span>
        <span class="{'profile-val' if dist else 'profile-val-notset'}">{_pv(dist)}</span>
      </div>
      <div class="profile-row">
        <span class="profile-lbl">💻 Branch</span>
        <span class="{'profile-val' if branch else 'profile-val-notset'}">{_pv(branch)}</span>
      </div>
      <div class="profile-row">
        <span class="profile-lbl">🏛️ College Type</span>
        <span class="{'profile-val' if ctype else 'profile-val-notset'}">{_pv(ctype)}</span>
      </div>
      {'<div class="profile-row"><span class="profile-lbl">🔍 Selected</span><span class="profile-val">' + sel_col + '</span></div>' if sel_col else ''}
    </div>
    <div style="text-align:center">
      <span class="status-badge {badge_cls}">{badge_dot} {badge_txt}</span>
    </div>
    """)

    if offset > 0:
        _html(f'<p style="text-align:center;font-size:0.72rem;color:var(--txt3);margin-bottom:0.5rem">Results shown: {offset}</p>')

    st.divider()

    # ── Control buttons ──────────────────────────────────────
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("🧹 Clear Filters", use_container_width=True,
                     help="Remove district / branch / type. Keeps cutoff & category."):
            clear_filters(state)
            st.rerun()
    with bc2:
        if st.button("🔄 Reset Chat", use_container_width=True,
                     help="Fully resets profile, history, and recommendations."):
            reset_state(state)
            st.session_state.messages     = []
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # ── Navigation / Quick Actions ───────────────────────────
    _html('<p class="sb-section">NAVIGATE</p>')

    nav_actions = [
        ("💬", "Chat",              None),
        ("🏫", "Top Colleges",      "top colleges in Tamil Nadu"),
        ("📍", "Colleges in Chennai","top colleges in Chennai"),
        ("⚖️", "Compare Colleges",  "compare Sri Venkateswara and SRM Valliammai"),
        ("📘", "TNEA Guide",         "what is TNEA?"),
        ("🎯", "My Options",         "show my college options"),
        ("💻", "CSE Colleges",       "top CSE colleges"),
        ("❓", "TNEA Cutoff Formula","how is TNEA cutoff calculated?"),
    ]

    for icon, label, msg in nav_actions:
        if msg is None:
            # "Chat" button — no action (already in chat)
            _html(f'<div class="sb-nav-btn" style="opacity:0.5;cursor:default">{icon} {label}</div>')
        else:
            if st.button(f"{icon} {label}", use_container_width=True, key=f"nav_{label}"):
                st.session_state._pending_message = msg
                st.rerun()

    st.divider()

    # ── Data Summary ─────────────────────────────────────────
    _html('<p class="sb-section">DATA SUMMARY</p>')

    data_status_dot = "dot-green" if _data_ok else "dot-red"
    data_status_txt = "Connected" if _data_ok else "Disconnected"

    _html(f"""
    <div class="data-card">
      <div class="data-row">
        <span class="data-key">📊 Total Records</span>
        <span class="data-val">{_total_recs:,}</span>
      </div>
      <div class="data-row">
        <span class="data-key">📍 Districts</span>
        <span class="data-val">{_total_dists}</span>
      </div>
      <div class="data-row">
        <span class="data-key">💻 Branches</span>
        <span class="data-val">{_total_branches}</span>
      </div>
      <div class="data-row">
        <span class="data-key">📅 Data Year</span>
        <span class="data-val">TNEA 2025</span>
      </div>
      <div class="data-row">
        <span class="data-key">🔗 Status</span>
        <span style="font-size:0.78rem;font-weight:600">
          <span class="dot-pulse {data_status_dot}" style="display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px"></span>
          <span style="color:{'#10b981' if _data_ok else '#ef4444'} !important">{data_status_txt}</span>
        </span>
      </div>
    </div>
    """)

    with st.expander("🔬 Validation Report"):
        st.markdown(validate_data())

    # ── Footer ───────────────────────────────────────────────
    _html('<div class="sb-footer">Made with ❤️ for TNEA Aspirants</div>')


# ═══════════════════════════════════════════════════════════════
# MAIN AREA — HEADER BAR
# ═══════════════════════════════════════════════════════════════

status_cls = "data-pill-green" if _data_ok else "data-pill-red"
status_dot = "dot-green"       if _data_ok else "dot-red"
status_txt = "TNEA 2025 Data Connected" if _data_ok else "Data Disconnected"

_html(f"""
<div class="saarthi-header">
  <div class="saarthi-header-left">
    <div class="saarthi-avatar-sm">🎓</div>
    <div>
      <p class="saarthi-brand-name">TNEA Saarthi</p>
      <p class="saarthi-brand-tag">Your AI College Counselling Companion</p>
    </div>
  </div>
  <div class="saarthi-header-right">
    <span class="data-pill {status_cls}">
      <span class="dot-pulse {status_dot}" style="width:7px;height:7px;border-radius:50%;display:inline-block"></span>
      {status_txt}
    </span>
  </div>
</div>
""")


# ═══════════════════════════════════════════════════════════════
# PENDING MESSAGE  (from sidebar buttons — must run BEFORE chat)
# ═══════════════════════════════════════════════════════════════

if st.session_state._pending_message is not None:
    pending = st.session_state._pending_message
    st.session_state._pending_message = None
    _process_and_display(pending)


# ═══════════════════════════════════════════════════════════════
# CHAT HISTORY DISPLAY
# ═══════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ═══════════════════════════════════════════════════════════════
# WELCOME CARD  (shown only when no conversation exists)
# ═══════════════════════════════════════════════════════════════

if not st.session_state.messages:

    # ── Visual welcome card ──────────────────────────────────
    _html("""
    <div class="welcome-card">
      <div class="welcome-avatar">🎓</div>
      <h2 class="welcome-title">👋 Vanakkam! I'm Saarthi</h2>
      <p class="welcome-sub">Your AI College Counselling Companion</p>
      <p class="welcome-desc">
        I help you find the best engineering colleges for TNEA 2025<br>
        based on your cutoff, community, district, and branch —<br>
        using real TNEA historical data.
      </p>
      <div class="welcome-chips">
        <span class="welcome-chip">🎯 TNEA Cutoff</span>
        <span class="welcome-chip">👤 Community</span>
        <span class="welcome-chip">📍 District</span>
        <span class="welcome-chip">💻 Branch</span>
        <span class="welcome-chip">🏛️ College Type</span>
      </div>
      <p style="font-size:0.75rem;color:var(--txt3);margin:0">Powered by real TNEA 2025 closing cutoff data</p>
    </div>
    """)

    # ── Quick action cards  (2 × 2 grid using columns) ───────
    _html('<p style="font-size:0.8rem;font-weight:600;color:var(--txt2);margin-bottom:0.5rem">What would you like to do?</p>')

    qc1, qc2 = st.columns(2)

    with qc1:
        if st.button("🎯 Personalized Recommendations", use_container_width=True, key="wq1"):
            st.session_state._pending_message = "show my college options"
            st.rerun()
        if st.button("⚖️ Compare Two Colleges", use_container_width=True, key="wq3"):
            st.session_state._pending_message = "compare Sri Venkateswara and SRM Valliammai"
            st.rerun()

    with qc2:
        if st.button("🏫 Top Colleges Discovery", use_container_width=True, key="wq2"):
            st.session_state._pending_message = "top colleges in Tamil Nadu"
            st.rerun()
        if st.button("❓ TNEA Help & Guide", use_container_width=True, key="wq4"):
            st.session_state._pending_message = "what is TNEA?"
            st.rerun()

    # ── Example inputs ────────────────────────────────────────
    _html("""
    <div style="margin-top:1.25rem; padding:1rem; background:rgba(255,255,255,0.03);
                border:1px solid rgba(255,255,255,0.06); border-radius:12px;">
      <p style="font-size:0.76rem; color:var(--txt3); margin:0 0 0.5rem; font-weight:600;
                text-transform:uppercase; letter-spacing:0.06em">Try typing...</p>
      <div style="display:flex; flex-wrap:wrap; gap:0.4rem">
        <code style="background:rgba(124,58,237,0.1); color:var(--purple-l);
                     padding:0.2rem 0.5rem; border-radius:5px; font-size:0.78rem">160 BC CSE Chennai</code>
        <code style="background:rgba(124,58,237,0.1); color:var(--purple-l);
                     padding:0.2rem 0.5rem; border-radius:5px; font-size:0.78rem">college code 2347</code>
        <code style="background:rgba(124,58,237,0.1); color:var(--purple-l);
                     padding:0.2rem 0.5rem; border-radius:5px; font-size:0.78rem">top colleges in Coimbatore</code>
        <code style="background:rgba(124,58,237,0.1); color:var(--purple-l);
                     padding:0.2rem 0.5rem; border-radius:5px; font-size:0.78rem">what is TNEA cutoff?</code>
      </div>
    </div>
    """)


# ═══════════════════════════════════════════════════════════════
# QUICK PROMPTS STRIP  (always visible above input)
# ═══════════════════════════════════════════════════════════════

_html('<div style="height:0.5rem"></div>')

quick_prompts = [
    ("🔍", "Find CSE Colleges",          "top CSE colleges"),
    ("📍", "Top Chennai Colleges",        "top colleges in Chennai"),
    ("⚖️", "Compare Colleges",           "compare Sri Venkateswara and SRM Valliammai"),
    ("📊", "Show My Options",             "show my college options"),
    ("❓", "TNEA Cutoff Formula",         "how is TNEA cutoff calculated?"),
    ("📘", "Counselling Process",         "explain TNEA counselling process"),
    ("🔁", "More Colleges",              "more"),
    ("🧹", "Clear Filters",              "clear filters"),
]

# Render as Streamlit buttons in two 4-column rows
row1_prompts = quick_prompts[:4]
row2_prompts = quick_prompts[4:]

r1_cols = st.columns(len(row1_prompts))
for col, (icon, label, msg) in zip(r1_cols, row1_prompts):
    with col:
        if st.button(f"{icon} {label}", use_container_width=True, key=f"qp_{label}"):
            st.session_state._pending_message = msg
            st.rerun()

r2_cols = st.columns(len(row2_prompts))
for col, (icon, label, msg) in zip(r2_cols, row2_prompts):
    with col:
        if st.button(f"{icon} {label}", use_container_width=True, key=f"qp_{label}"):
            st.session_state._pending_message = msg
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# SMART CHAT INPUT
# ═══════════════════════════════════════════════════════════════

user_message = st.chat_input(
    "Ask Saarthi anything about TNEA colleges... "
    "(e.g. 160 BC CSE Chennai  |  compare SVCE and SRM  |  what is TNEA?)"
)

if user_message:
    _process_and_display(user_message)
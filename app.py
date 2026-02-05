import streamlit as st
import pandas as pd
import requests
import time
from typing import Optional

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="PaperPicker",
    page_icon="📄",
    layout="wide",
)

S2_BASE = "https://api.semanticscholar.org/graph/v1"

# -----------------------------
# Semantic Scholar search (with 429 backoff)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def semantic_scholar_search(
    query: str,
    limit: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
):
    if not query.strip():
        return pd.DataFrame()

    fields = ",".join([
        "paperId", "title", "year", "venue", "authors",
        "abstract", "url", "citationCount", "referenceCount"
    ])

    params = {
        "query": query,
        "limit": int(limit),
        "fields": fields,
    }

    last_err = None
    papers = []
    for attempt in range(4):
        try:
            r = requests.get(f"{S2_BASE}/paper/search", params=params, timeout=30)

            if r.status_code == 429:
                time.sleep(2 + attempt)
                continue

            r.raise_for_status()
            data = r.json()
            papers = data.get("data", [])
            break
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    else:
        raise Exception(f"Semantic Scholar request failed (likely rate limited). Last error: {last_err}")

    rows = []
    for p in papers:
        authors = ", ".join([a.get("name", "") for a in p.get("authors", [])][:6])
        year = p.get("year", None)

        rows.append({
            "paper_id": p.get("paperId"),
            "title": p.get("title", ""),
            "year": year,
            "venue": p.get("venue", ""),
            "authors": authors,
            "abstract": p.get("abstract", "") or "",
            "url": p.get("url", "") or "",
            "citationCount": p.get("citationCount", 0),
            "referenceCount": p.get("referenceCount", 0),
        })

    df = pd.DataFrame(rows)

    if not df.empty and year_from is not None:
        df = df[df["year"].fillna(0) >= year_from]
    if not df.empty and year_to is not None:
        df = df[df["year"].fillna(9999) <= year_to]

    if "paper_id" in df.columns:
        df = df.drop_duplicates(subset=["paper_id"])

    return df.reset_index(drop=True)

# -----------------------------
# Session state
# -----------------------------
def init_state():
    if "mode" not in st.session_state:
        st.session_state.mode = "Shop"  # Shop vs Swipe
    
    if "pending_mode" not in st.session_state:
        st.session_state.pending_mode = st.session_state.mode  # mirrors the selectbox

    if "confirm_mode_switch" not in st.session_state:
        st.session_state.confirm_mode_switch = False

    if "reading_list" not in st.session_state:
        st.session_state.reading_list = []  # list of paper_id

    if "papers" not in st.session_state:
        st.session_state.papers = pd.DataFrame()

    if "swipe_index" not in st.session_state:
        st.session_state.swipe_index = 0

    if "user_query" not in st.session_state:
        st.session_state.user_query = ""

    # "query" -> enter query; "fetch" -> fetch & browse
    if "step" not in st.session_state:
        st.session_state.step = "query"

    if "final_query" not in st.session_state:
        st.session_state.final_query = ""

    if "page" not in st.session_state:
        st.session_state.page = 0

    # Tracks decisions in swipe mode: paper_id -> "read" | "skip"
    if "swipe_decisions" not in st.session_state:
        st.session_state.swipe_decisions = {}

    # Cache for AI summaries: paper_id -> summary text
    if "ai_summaries" not in st.session_state:
        st.session_state.ai_summaries = {}  # paper_id -> summary text


init_state()

def add_to_reading_list(paper_id: str):
    if paper_id and paper_id not in st.session_state.reading_list:
        st.session_state.reading_list.append(paper_id)

def remove_from_reading_list(paper_id: str):
    st.session_state.reading_list = [pid for pid in st.session_state.reading_list if pid != paper_id]

def toggle_reading_list(paper_id: str):
    if paper_id in st.session_state.reading_list:
        remove_from_reading_list(paper_id)
    else:
        add_to_reading_list(paper_id)

def reading_list_df():
    df = st.session_state.papers
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df["paper_id"].isin(st.session_state.reading_list)].copy()

# -----------------------------
# Header (toggle in top-right)
# -----------------------------
header_left, header_right = st.columns([3, 1])
def _reset_for_mode_switch():
    st.session_state.reading_list = []
    st.session_state.swipe_decisions = {}
    st.session_state.swipe_index = 0
    st.session_state.page = 0

def on_mode_select_change():
    if st.session_state.pending_mode != st.session_state.mode:
        st.session_state.confirm_mode_switch = True

with header_right:
    st.selectbox(
        "Mode",
        ["Shop", "Swipe"],
        key="pending_mode",
        on_change=on_mode_select_change,
    )
    st.metric("Reading List", len(st.session_state.reading_list))

@st.dialog("Confirm mode switch")
def confirm_mode_dialog():
    new_mode = st.session_state.pending_mode
    st.write(f'You are switching to **{new_mode}** mode. Your reading list will clear.')
    c1, c2 = st.columns(2)

    with c1:
        if st.button("✅ Yes, switch", use_container_width=True):
            st.session_state.mode = new_mode
            _reset_for_mode_switch()
            st.session_state.confirm_mode_switch = False
            st.rerun()

    with c2:
        if st.button("❌ No, keep current", use_container_width=True):
            # revert the dropdown selection back to the current mode
            st.session_state.pending_mode = st.session_state.mode
            st.session_state.confirm_mode_switch = False
            st.rerun()

if st.session_state.confirm_mode_switch:
    confirm_mode_dialog()

with header_left:
    st.title("📄 PaperPicker")
    st.caption("Search → Browse (Shop/Swipe) → Read")

# -----------------------------
# Tabs
# -----------------------------
tab_search, tab_list = st.tabs(["Search", "Reading List"])

# -----------------------------
# Shared: Paper card renderer
#   NOTE: widget_key_prefix prevents DuplicateWidgetID across tabs/views
# -----------------------------
def render_paper_card(row, in_shop_context: bool, widget_key_prefix: str, show_actions: bool = True):
    paper_id = row["paper_id"]
    in_list = paper_id in st.session_state.reading_list

    st.write(f"**{row['title']}**")
    meta = f"{row.get('authors','')} • {row.get('venue','')} • {row.get('year','')}"
    st.caption(meta)

    meta = f"{row.get('authors','')} • {row.get('venue','')} • {row.get('year','')}"

    # AI Summary (on-demand)
    if not GEMINI_API_KEY:
        st.caption("🔒 Add GEMINI_API_KEY to .streamlit/secrets.toml to enable AI summaries.")
    else:
        # show cached/in-session summary if already generated
        if paper_id in st.session_state.ai_summaries:
            st.markdown(st.session_state.ai_summaries[paper_id])
        else:
            if st.button("✨ Summarize with AI", key=f"{widget_key_prefix}_sum_{paper_id}"):
                with st.spinner("Generating summary..."):
                    try:
                        summary = gemini_summary(
                            title=row.get("title", ""),
                            abstract=row.get("abstract", "") or "",
                            meta=meta,
                        )
                        st.session_state.ai_summaries[paper_id] = summary
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gemini summary failed: {e}")


    with st.expander("Abstract / Link"):
        st.write(row.get("abstract", "") or "No abstract.")
        url = row.get("url", "")
        if url:
            st.link_button("Open paper", url)

    if not show_actions:
        return

    if in_shop_context:
        label = "🗑️ Remove from reading list" if in_list else "➕ Add to reading list"
        if st.button(label, key=f"{widget_key_prefix}_toggle_{paper_id}"):
            toggle_reading_list(paper_id)
            st.rerun()

# -----------------------------
# Helper: Swipe UI
# -----------------------------
def render_swipe_ui(df: pd.DataFrame, context_key: str):
    idx = st.session_state.swipe_index
    if idx >= len(df):
        st.success("You’ve seen all papers in this pool.")
        return

    row = df.iloc[idx]
    pid = row["paper_id"]
    decision = st.session_state.swipe_decisions.get(pid, None)

    with st.container(border=True):
        render_paper_card(
            row,
            in_shop_context=False,
            widget_key_prefix=f"{context_key}_swipe_{pid}",
            show_actions=False
        )

    # Skip on LEFT, Read on RIGHT
    left, spacer, right = st.columns([1, 6, 1])

    read_class = "swipe-btn-read active" if decision == "read" else "swipe-btn-read"
    skip_class = "swipe-btn-skip active" if decision == "skip" else "swipe-btn-skip"

    with left:
        st.markdown(f'<div class="{skip_class}">', unsafe_allow_html=True)
        if st.button("❌ Skip", key=f"{context_key}_skip_{pid}", use_container_width=True):
            remove_from_reading_list(pid)
            st.session_state.swipe_decisions[pid] = "skip"
            st.session_state.swipe_index += 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(f'<div class="{read_class}">', unsafe_allow_html=True)
        if st.button("📖 To be read", key=f"{context_key}_read_{pid}", use_container_width=True):
            add_to_reading_list(pid)
            st.session_state.swipe_decisions[pid] = "read"
            st.session_state.swipe_index += 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.progress((idx + 1) / len(df))
    st.caption(f"Paper {idx + 1} of {len(df)}")

# -----------------------------
# Helper: Shop UI
# -----------------------------
def render_shop_ui(df: pd.DataFrame, context_key: str):
    page_size = 10
    total = len(df)
    max_page = max(0, (total - 1) // page_size)

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("⬅ Prev", key=f"{context_key}_prev", disabled=(st.session_state.page <= 0)):
            st.session_state.page -= 1
            st.rerun()
    with nav2:
        st.write(f"Page {st.session_state.page + 1} of {max_page + 1}")
    with nav3:
        if st.button("Next ➜", key=f"{context_key}_next", disabled=(st.session_state.page >= max_page)):
            st.session_state.page += 1
            st.rerun()

    start = st.session_state.page * page_size
    end = min(start + page_size, total)
    view = df.iloc[start:end]

    cols = st.columns(2)
    for i, row in view.iterrows():
        with cols[i % 2]:
            with st.container(border=True):
                render_paper_card(row, in_shop_context=True, widget_key_prefix=f"{context_key}_shop", show_actions=True)


# -----------------------------
# Gemini (AI summaries)
# -----------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"  # example model id shown in Gemini API reference :contentReference[oaicite:1]{index=1}

def _extract_gemini_text(resp_json: dict) -> str:
    # Typical shape: candidates[0].content.parts[*].text :contentReference[oaicite:2]{index=2}
    try:
        parts = resp_json["candidates"][0]["content"]["parts"]
        return "\n".join([p.get("text", "") for p in parts]).strip()
    except Exception:
        return ""

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)  # cache summaries for 24h
def gemini_summary(title: str, abstract: str, meta: str) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ Missing GEMINI_API_KEY in .streamlit/secrets.toml"

    if not (title.strip() or abstract.strip()):
        return "No text available to summarize."

    prompt = f"""
Based ONLY on the title + abstract below, write a concise 2-4 sentence paragraph
summarizing the paper's main contributions and key results. No bullets.
Do not add facts that are not in the title/abstract.

Title: {title}

Meta: {meta}

Abstract:
{abstract}
""".strip()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        # Optional: keep output short/consistent
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 950,
        },
    }

    r = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,  # per docs :contentReference[oaicite:3]{index=3}
        },
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    text = _extract_gemini_text(r.json())
    return text or "⚠️ Gemini returned an empty response."


# -----------------------------
# Tab 1: Search Funnel
# -----------------------------
with tab_search:
    if st.session_state.step == "query":
        st.write("### Search")
        with st.form("query_form"):
            q = st.text_input(
                "Enter a search query",
                value=st.session_state.user_query,
                placeholder="e.g., human-ai interaction in educational technology",
            )
            submitted = st.form_submit_button("Next ➜")

        if submitted:
            st.session_state.user_query = q.strip()
            if not st.session_state.user_query:
                st.warning("Please enter a query.")
            else:
                st.session_state.final_query = st.session_state.user_query
                st.session_state.step = "fetch"
                st.rerun()

    else:
        st.write("### Fetch papers")
        st.caption(f"Query: **{st.session_state.final_query or st.session_state.user_query}**")

        with st.form("fetch_form"):
            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                limit = st.slider("How many papers (pool size)", 5, 20, 10)
            with colB:
                year_from = st.number_input("Year from", value=2018, step=1)
            with colC:
                year_to = st.number_input("Year to", value=2026, step=1)

            # --- buttons on the same row ---
            b1, b2, _sp = st.columns([1, 1, 6])
            with b1:
                fetched = st.form_submit_button("🔎 Fetch papers", use_container_width=True)
            with b2:
                start_over_clicked = st.form_submit_button("🔄 Start over", use_container_width=True)

        # Handle Start over
        if start_over_clicked:
            st.session_state.step = "query"
            st.session_state.user_query = ""
            st.session_state.final_query = ""
            st.session_state.papers = pd.DataFrame()
            st.session_state.page = 0
            st.session_state.swipe_index = 0
            st.session_state.swipe_decisions = {}
            # optional: keep reading list or clear it — your call
            # st.session_state.reading_list = []
            st.rerun()

        # Handle Fetch
        if fetched:
            with st.spinner("Searching Semantic Scholar..."):
                try:
                    df = semantic_scholar_search(
                        st.session_state.final_query or st.session_state.user_query,
                        limit=limit,
                        year_from=int(year_from),
                        year_to=int(year_to),
                    )
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    df = pd.DataFrame()

            st.session_state.papers = df
            st.session_state.swipe_index = 0
            st.session_state.page = 0
            st.session_state.swipe_decisions = {}
            st.toast(f"Loaded {len(df)} papers.")
            st.rerun()


        df = st.session_state.papers
        if df is None or df.empty:
            st.info("No papers loaded yet. Click **Fetch papers**.")
        else:
            st.caption("Results (browse here):")
            if st.session_state.mode == "Shop":
                render_shop_ui(df, context_key="search")
            else:
                render_swipe_ui(df, context_key="search")

# -----------------------------
# Tab 3: Reading List
# -----------------------------
with tab_list:
    st.subheader("Reading List")

    rl = reading_list_df()
    if rl.empty:
        st.info("Your reading list is empty. Add papers from the Search tab.")
    else:
        for _, row in rl.iterrows():
            pid = row["paper_id"]
            with st.container(border=True):
                st.write(f"**{row['title']}**")
                st.caption(f"{row.get('authors','')} • {row.get('venue','')} • {row.get('year','')}")
                c1, c2 = st.columns([1, 3])
                with c1:
                    if st.button("🗑️ Remove", key=f"rm_{pid}"):
                        remove_from_reading_list(pid)
                        st.rerun()
                with c2:
                    url = row.get("url", "")
                    if url:
                        st.link_button("Open paper", url)

        st.divider()
        st.download_button(
            "⬇️ Download reading list (CSV)",
            data=rl.to_csv(index=False).encode("utf-8"),
            file_name="reading_list.csv",
            mime="text/csv",
        )

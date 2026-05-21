"""
app.py — IMDB Sentiment Analyzer · Cinematic Dark UI
Run: streamlit run app.py

Requirements:
    pip install streamlit torch transformers

Model path: ./sentiment_model
    (fine-tuned distilbert-base-uncased, saved via model.save_pretrained())
"""

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & root variables ── */
:root {
    --bg:            #0a0a0f;
    --surface:       #111118;
    --surface2:      #16161f;
    --border:        rgba(255,255,255,0.07);
    --border-glow:   rgba(245,200,66,0.30);
    --gold:          #f5c842;
    --gold-dim:      #a8882a;
    --text:          #f0ede6;
    --text-muted:    #6b6978;
    --text-soft:     #9d9aaa;
    --positive:      #2dbd85;
    --positive-bg:   #0d2a1e;
    --positive-glow: rgba(45,189,133,0.12);
    --negative:      #e05252;
    --negative-bg:   #2a0d0d;
    --negative-glow: rgba(224,82,82,0.12);
}

/* Global overrides */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background-color: var(--bg) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}

/* ── Film strip decoration ── */
.film-strip {
    width: 100%;
    height: 6px;
    background: repeating-linear-gradient(
        90deg,
        var(--gold) 0px, var(--gold) 14px,
        transparent 14px, transparent 22px
    );
    opacity: 0.5;
    margin-bottom: 0;
}

/* ── Header ── */
.sa-header {
    text-align: center;
    padding: 44px 24px 36px;
    animation: fadeDown 0.55s ease both;
}

.sa-logo-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-bottom: 6px;
}

.sa-logo-icon {
    width: 46px;
    height: 46px;
    border-radius: 10px;
    background: var(--gold);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}

.sa-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 52px !important;
    letter-spacing: 3px !important;
    color: var(--text) !important;
    line-height: 1 !important;
    margin: 0 !important;
}

.sa-title span { color: var(--gold); }

.sa-subtitle {
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}

.sa-subtitle::before, .sa-subtitle::after {
    content: '';
    display: block;
    height: 1px;
    width: 40px;
    background: var(--border);
}

/* ── Card ── */
.sa-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 28px;
    margin: 0 16px 16px;
    animation: fadeUp 0.5s ease both;
}

.sa-card-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 14px;
    font-weight: 500;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    caret-color: var(--gold) !important;
    transition: border-color 0.2s !important;
}

.stTextArea textarea:focus {
    border-color: rgba(245,200,66,0.35) !important;
    box-shadow: none !important;
}

.stTextArea textarea::placeholder { color: var(--text-muted) !important; }
.stTextArea label { display: none !important; }

/* ── Buttons (general) ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-soft) !important;
    padding: 8px 16px !important;
}

.stButton > button:hover {
    border-color: var(--border-glow) !important;
    color: var(--text) !important;
    background: rgba(245,200,66,0.05) !important;
}

/* Primary (Analyze) button */
.stButton > button[kind="primary"] {
    width: 100% !important;
    background: var(--gold) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 20px !important;
    letter-spacing: 2px !important;
    color: #0a0a0f !important;
    padding: 16px !important;
    transition: all 0.2s !important;
}

.stButton > button[kind="primary"]:hover {
    background: #f7d35a !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ── Result panels ── */
.result-positive {
    background: var(--positive-bg);
    border: 1px solid rgba(45,189,133,0.25);
    border-radius: 16px;
    padding: 28px;
    margin: 0 16px 16px;
    box-shadow: 0 0 40px var(--positive-glow);
    animation: fadeUp 0.4s ease both;
}

.result-negative {
    background: var(--negative-bg);
    border: 1px solid rgba(224,82,82,0.25);
    border-radius: 16px;
    padding: 28px;
    margin: 0 16px 16px;
    box-shadow: 0 0 40px var(--negative-glow);
    animation: fadeUp 0.4s ease both;
}

.verdict-row {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 22px;
}

.verdict-icon {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 500;
    flex-shrink: 0;
}

.verdict-icon.pos {
    background: var(--positive-glow);
    color: var(--positive);
}

.verdict-icon.neg {
    background: var(--negative-glow);
    color: var(--negative);
}

.verdict-label {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 38px;
    letter-spacing: 2px;
    line-height: 1;
}

.verdict-conf {
    font-size: 13px;
    font-family: 'DM Mono', monospace;
    margin-top: 3px;
    opacity: 0.7;
}

.verdict-label.pos { color: var(--positive); }
.verdict-label.neg { color: var(--negative); }
.verdict-conf.pos  { color: var(--positive); }
.verdict-conf.neg  { color: var(--negative); }

/* ── Score bars ── */
.score-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 0 0 20px;
}

.score-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 10px;
}

.score-row:last-child { margin-bottom: 0; }

.score-lbl {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    width: 56px;
    flex-shrink: 0;
}

.score-lbl.pos { color: var(--positive); opacity: 0.85; }
.score-lbl.neg { color: var(--negative); opacity: 0.85; }

.bar-track {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.16,1,0.3,1);
}

.bar-fill.pos { background: var(--positive); }
.bar-fill.neg { background: var(--negative); }

.score-pct {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}

.score-pct.pos { color: var(--positive); }
.score-pct.neg { color: var(--negative); }

/* ── Model info expander ── */
details {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    margin: 0 16px 0 !important;
    overflow: hidden !important;
}

details summary {
    font-size: 12px !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: var(--text-soft) !important;
    padding: 16px 24px !important;
    cursor: pointer !important;
    list-style: none !important;
    display: flex !important;
    justify-content: space-between !important;
}

details summary:hover { color: var(--text) !important; }

.model-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border-top: 1px solid var(--border);
}

.model-cell {
    background: var(--surface);
    padding: 14px 20px;
}

.model-cell-lbl {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.model-cell-val {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: var(--text);
}

.model-cell-val.gold { color: var(--gold); }

/* ── Warning banner ── */
.warn-banner {
    background: rgba(245,200,66,0.06);
    border: 1px solid rgba(245,200,66,0.20);
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 13px;
    color: var(--gold);
    margin: 0 16px 16px;
    text-align: center;
}

/* ── Spinner override ── */
.stSpinner > div {
    border-top-color: var(--gold) !important;
}

/* ── Keyframes ── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "./sentiment_model"
MAX_LEN    = 256

EXAMPLES = {
    "👍 Positive": (
        "An absolute masterpiece. The performances are riveting from start to finish — "
        "every frame deliberate, every line earned. This is cinema at its most powerful "
        "and humane. I was moved to tears."
    ),
    "👎 Negative": (
        "Completely unwatchable. The plot makes no sense, the acting is wooden, and "
        "the CGI looks like it was rendered on a toaster. Save yourself two hours "
        "and watch paint dry instead."
    ),
    "🤔 Mixed": (
        "The visuals are stunning and the lead gives a career-best performance, but "
        "the script is a mess. Pacing drags badly in the second act and the ending "
        "felt completely unearned. Worth a rental, not a cinema ticket."
    ),
}

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(text: str, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).numpy()[0]
    idx   = int(np.argmax(probs))
    label = ["NEGATIVE", "POSITIVE"][idx]
    return label, float(probs[idx]), float(probs[0]), float(probs[1])

# ── Session state ─────────────────────────────────────────────────────────────
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# ── Film strip top ────────────────────────────────────────────────────────────
st.markdown('<div class="film-strip"></div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sa-header">
  <div class="sa-logo-row">
    <div class="sa-logo-icon">🎬</div>
    <h1 class="sa-title">SENTI<span>MENT</span></h1>
  </div>
  <p class="sa-subtitle">DistilBERT &nbsp;·&nbsp; IMDB &nbsp;·&nbsp; 25K Reviews</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    tokenizer, model = load_model()

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="sa-card"><div class="sa-card-label">Review</div>', unsafe_allow_html=True)

review = st.text_area(
    label="review",
    value=st.session_state.review_text,
    placeholder="Paste a movie review…",
    height=150,
    label_visibility="collapsed",
    key="review_input",
)

char_count = len(review)
st.markdown(
    f'<div style="text-align:right;font-size:12px;color:var(--text-muted);'
    f'font-family:DM Mono,monospace;margin-top:6px">{char_count} chars</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Example buttons card ──────────────────────────────────────────────────────
st.markdown('<div class="sa-card" style="padding:20px 28px">'
            '<div class="sa-card-label">Try an example</div>', unsafe_allow_html=True)

ex_cols = st.columns(len(EXAMPLES))
for col, (label_ex, text_ex) in zip(ex_cols, EXAMPLES.items()):
    if col.button(label_ex, key=f"ex_{label_ex}"):
        st.session_state.review_text = text_ex
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ── Analyze button ────────────────────────────────────────────────────────────
st.markdown('<div style="margin:0 16px 16px">', unsafe_allow_html=True)
run = st.button("Analyze", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Run inference ─────────────────────────────────────────────────────────────
if run:
    text_to_analyze = st.session_state.get("review_input", review).strip()

    if not text_to_analyze:
        st.markdown(
            '<div class="warn-banner">Please enter a review before analyzing.</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analyzing…"):
            label, conf, p_neg, p_pos = predict(text_to_analyze, tokenizer, model)

        is_pos     = label == "POSITIVE"
        cls        = "positive" if is_pos else "negative"
        icon_sym   = "✦" if is_pos else "✕"
        icon_cls   = "pos" if is_pos else "neg"
        lbl_cls    = "pos" if is_pos else "neg"

        st.markdown(f"""
        <div class="result-{cls}">
          <div class="verdict-row">
            <div class="verdict-icon {icon_cls}">{icon_sym}</div>
            <div>
              <div class="verdict-label {lbl_cls}">{label}</div>
              <div class="verdict-conf {lbl_cls}">{conf*100:.1f}% confident</div>
            </div>
          </div>
          <div class="score-divider"></div>
          <div class="score-row">
            <div class="score-lbl pos">Pos</div>
            <div class="bar-track">
              <div class="bar-fill pos" style="width:{p_pos*100:.1f}%"></div>
            </div>
            <div class="score-pct pos">{p_pos*100:.1f}%</div>
          </div>
          <div class="score-row">
            <div class="score-lbl neg">Neg</div>
            <div class="bar-track">
              <div class="bar-fill neg" style="width:{p_neg*100:.1f}%"></div>
            </div>
            <div class="score-pct neg">{p_neg*100:.1f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Model details expander ────────────────────────────────────────────────────
st.markdown("""
<details>
  <summary>Model Details ▾</summary>
  <div class="model-grid">
    <div class="model-cell">
      <div class="model-cell-lbl">Base model</div>
      <div class="model-cell-val">distilbert-base-uncased</div>
    </div>
    <div class="model-cell">
      <div class="model-cell-lbl">Dataset</div>
      <div class="model-cell-val">IMDB 25K / 25K</div>
    </div>
    <div class="model-cell">
      <div class="model-cell-lbl">Max tokens</div>
      <div class="model-cell-val">256</div>
    </div>
    <div class="model-cell">
      <div class="model-cell-lbl">Epochs</div>
      <div class="model-cell-val">3</div>
    </div>
    <div class="model-cell">
      <div class="model-cell-lbl">Learning rate</div>
      <div class="model-cell-val">2e-5</div>
    </div>
    <div class="model-cell">
      <div class="model-cell-lbl">Accuracy</div>
      <div class="model-cell-val gold">89.7%</div>
    </div>
    <div class="model-cell">
      <div class="model-cell-lbl">F1 Score</div>
      <div class="model-cell-val gold">0.897</div>
    </div>
    <div class="model-cell">
      <div class="model-cell-lbl">Framework</div>
      <div class="model-cell-val">HuggingFace</div>
    </div>
  </div>
</details>
<div style="height:48px"></div>
<div class="film-strip" style="margin-top:auto"></div>
""", unsafe_allow_html=True)
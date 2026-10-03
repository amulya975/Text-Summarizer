import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)


# -----------------------------
# Models
# -----------------------------
MODELS = {
    "DistilBART (faster, lighter)": "chunchuamulya03/distilbart-cnn-12-6",
    "BART Large (better quality)": "facebook/bart-large-cnn",
}



# -----------------------------
# Summary lengths
# -----------------------------
LENGTHS = {
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250),
}


CHUNK_WORDS = 600


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource(show_spinner="Loading model (first run downloads it)...")
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    return tokenizer, model


# -----------------------------
# Split text into chunks
# -----------------------------
def chunk_text(text: str, size: int = CHUNK_WORDS):
    words = text.split()

    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]


# -----------------------------
# Summarize one chunk
# -----------------------------
def summarize_chunks(text, tokenizer, model, min_len, max_len):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    ids = model.generate(
        **inputs,
        min_length=min_len,
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    return tokenizer.decode(
        ids[0],
        skip_special_tokens=True
    )


# -----------------------------
# Summarize complete text
# -----------------------------
def summarize(text, tokenizer, model, min_len, max_len):
    chunks = chunk_text(text)

    progress = st.progress(
        0.0,
        text="Summarizing..."
    )

    parts = []

    for i, chunk in enumerate(chunks, 1):
        words = len(chunk.split())

        chunk_min_len = min(min_len, max(10, words // 2))

        parts.append(
            summarize_chunks(
                chunk,
                tokenizer,
                model,
                chunk_min_len,
                max_len
            )
        )

        progress.progress(
            i / len(chunks),
            text=f"Summarized chunk {i}/{len(chunks)}"
        )

    progress.empty()

    combined = " ".join(parts)

    # If there are multiple chunks, summarize the combined summaries
    if len(chunks) > 1 and len(combined.split()) > CHUNK_WORDS:
        combined = summarize_chunks(
            combined,
            tokenizer,
            model,
            min_len,
            max_len * 2
        )

    return combined


# -----------------------------
# Read PDF
# -----------------------------
def read_pdf(file):
    reader = PdfReader(file)

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

model_name = st.sidebar.selectbox(
    "Model",
    list(MODELS.keys())
)

length = st.sidebar.radio(
    "Summary length",
    list(LENGTHS.keys()),
    index=1
)


# -----------------------------
# Main application
# -----------------------------
st.title("📝 Text Summarizer")

st.write(
    "Paste text or upload a file (.txt / .pdf) "
    "to get a concise summary."
)


# -----------------------------
# Input selection
# -----------------------------
source = st.radio(
    "Input type",
    ["Paste text", "Upload file"],
    horizontal=True
)

text = ""


if source == "Paste text":

    text = st.text_area(
        "Your text",
        height=300,
        placeholder="Paste an article, notes, or report here..."
    )

else:

    file = st.file_uploader(
        "Upload .txt or .pdf",
        type=["txt", "pdf"]
    )

    if file:

        if file.name.lower().endswith(".pdf"):
            text = read_pdf(file)

        else:
            text = file.read().decode(
                "utf-8",
                errors="ignore"
            )

        with st.expander("Preview extracted text"):
            st.write(
                text[:3000]
                + ("..." if len(text) > 3000 else "")
            )


# -----------------------------
# Summarize button
# -----------------------------
if st.button("✨ Summarize", type="primary"):

    word_count = len(text.split())

    if word_count < 50:

        st.warning(
            "Please provide at least 50 words of text."
        )

    else:

        tokenizer, model = load_model(
            MODELS[model_name]
        )

        min_len, max_len = LENGTHS[length]

        summary = summarize(
            text,
            tokenizer,
            model,
            min_len,
            max_len
        )

        col1, col2 = st.columns(2)

        # Original text
        with col1:

            st.subheader("Original")

            st.caption(
                f"{word_count} words"
            )

            st.write(
                text[:5000]
                + ("..." if len(text) > 5000 else "")
            )

        # Summary
        with col2:

            st.subheader("Summary")

            summary_word_count = len(
                summary.split()
            )

            reduction = max(
                0,
                100 - (
                    summary_word_count * 100 // word_count
                )
            )

            st.caption(
                f"{summary_word_count} words • "
                f"{reduction}% shorter"
            )

            st.success(summary)

            st.download_button(
                "⬇️ Download summary",
                summary,
                "summary.txt",
                "text/plain"
            )

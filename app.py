import os
import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import string
import nltk

# Optional: ensure stopwords available if review_cleaning is referenced during unpickling
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords')
    except Exception:
        pass

# Define the same function name that may have been used when pickling the vectorizer
# Keep implementation consistent to avoid unpickle errors
# Note: This intentionally mirrors the original (including its quirks)
def review_cleaning(review):
    punch_remover = [char for char in review if char not in string.punctuation]
    puch_remover_join = ''.join(punch_remover)
    cleaned_puch = [
        words for words in puch_remover_join.split()
        if words.lower not in nltk.corpus.stopwords.words('english')  # noqa: E712 - mirrors original
    ]
    return cleaned_puch

@st.cache_resource(show_spinner=False)
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vec_path = os.path.join(base_dir, 'vectorizer.pkl')
    clf_path = os.path.join(base_dir, 'classifier.pkl')
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(clf_path, 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_artifacts()

LABEL_MAP = {0: 'Negative', 1: 'Positive'}

# Simple lexicon for extremely short inputs (heuristic)
_POS_WORDS = {
    'good','great','excellent','amazing','awesome','fantastic','love','loved','like','liked','perfect','nice','wonderful','brilliant','superb','positive','satisfied','happy','best'
}
_NEG_WORDS = {
    'bad','terrible','awful','hate','hated','poor','worst','disappointed','disappointing','broken','useless','garbage','horrible','buggy','noisy','cheap','negative','unhappy'
}
_NEGATIONS = {'not','no','never','hardly','scarcely','barely','without'}


def _heuristic_short_text_label(text: str) -> int | None:
    # Returns 1 (Positive), 0 (Negative) or None if uncertain
    t = ''.join(ch for ch in text.lower() if ch not in string.punctuation)
    tokens = t.split()
    if len(tokens) == 0:
        return None
    pos = sum(tok in _POS_WORDS for tok in tokens)
    neg = sum(tok in _NEG_WORDS for tok in tokens)
    has_negation = any(tok in _NEGATIONS for tok in tokens)
    if has_negation and pos > 0 and neg == 0:
        return 0
    # common phrase boost
    if 'too good' in t and not has_negation:
        return 1
    score = pos - neg
    if len(tokens) <= 5 and abs(score) >= 1:
        return 1 if score > 0 else 0
    return None

# Predict helper to ensure identical preprocessing pipeline for single and batch
def predict_labels(raw_texts: list[str]) -> np.ndarray:
    # mirror batch path exactly: pandas Series -> astype(str) -> fillna -> vectorizer.transform
    s = pd.Series(raw_texts).astype(str).fillna('')
    Xb = vectorizer.transform(s.values)
    return model.predict(Xb).astype(int)

# ---------- UI THEME AND HEADER ----------
st.set_page_config(page_title='Amazon Reviews Sentiment', page_icon='🛍️', layout='centered')

st.markdown(
    """
    <style>
      .sentiment-badge { padding:6px 12px; border-radius:999px; font-weight:600; display:inline-block; }
      .sentiment-pos { background:#eafaf1; color:#1e8449; border:1px solid #1e844920; }
      .sentiment-neg { background:#fdecea; color:#c0392b; border:1px solid #c0392b20; }
      .subtitle { color:#6c757d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('🛍️ Amazon Reviews Sentiment Analysis')
st.caption('Predict sentiment for a single review or upload a CSV for batch prediction. ⚡')

with st.sidebar:
    st.header('✨ About')
    st.write('This app uses a trained model to classify Amazon reviews as Positive or Negative.')
    st.markdown('- 🔎 Single review analysis\n- 📄 Batch CSV analysis\n- 🥧 Distribution chart\n- ⬇️ Results download (with reviews labled as Positive & Negative)')
    st.success('Model loaded ✅')

# Tabs for sections
single_tab, batch_tab = st.tabs(["✍️ Single Review Analysis", "📄 Batch Review Analysis"])

# ---------- Single Review Analysis ----------
with single_tab:
    st.write('Type or paste a review and click Analyze to predict its sentiment.')
    single_text = st.text_area('📝 Review text', height=160, placeholder='e.g., “Absolutely love this product! Great battery life and build quality.”')
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_single = st.button('Analyze 🔎')

    if analyze_single:
        txt = (single_text or '').strip()
        if not txt:
            st.warning('⚠️ Please enter a review text.')
        else:
            # Heuristic for very short texts to fix obvious cases like "good"
            heuristic = _heuristic_short_text_label(txt)
            if heuristic is not None:
                pred = int(heuristic)
            else:
                preds = predict_labels([txt])
                pred = int(preds[0])
            label = LABEL_MAP.get(pred, str(pred))

            # Confidence (if available)
            conf = None
            if hasattr(model, 'predict_proba'):
                try:
                    # Recompute features for proba using the same path
                    s = pd.Series([txt]).astype(str).fillna('')
                    X = vectorizer.transform(s.values)
                    proba = model.predict_proba(X)[0]
                    conf = float(np.max(proba))
                except Exception:
                    conf = None

            if pred == 1:
                st.success('Sentiment: Positive 😊')
                st.markdown('<span class="sentiment-badge sentiment-pos">Positive</span>', unsafe_allow_html=True)
            else:
                st.error('Sentiment: Negative 😡')
                st.markdown('<span class="sentiment-badge sentiment-neg">Negative</span>', unsafe_allow_html=True)

            if conf is not None:
                st.progress(min(max(conf, 0.0), 1.0))
                st.caption(f'Confidence: {conf:.1%}')

            if hasattr(st, 'toast'):
                st.toast('Single review analyzed 🎯', icon='✅')

# ---------- Batch Review Analysis ----------
with batch_tab:
    st.write('Upload a CSV with one column of text reviews. Choose the column and run batch analysis.')
    sample = pd.DataFrame({'review': [
        'Great product, works as expected!',
        'Terrible quality. Broke after 2 days.'
    ]})
    csv_sample = io.StringIO()
    sample.to_csv(csv_sample, index=False)

    st.download_button('📥 Download CSV template', data=csv_sample.getvalue(), file_name='reviews_template.csv', mime='text/csv')

    uploaded = st.file_uploader('📄 Upload CSV', type=['csv'], help='CSV should have a single column with text reviews. You can select the column after upload.')

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding_errors='ignore')

        if df.shape[1] == 0:
            st.error('❗ No columns found in the uploaded CSV.')
        else:
            default_col = df.columns[0]
            text_col = st.selectbox('🧭 Select the review text column', options=list(df.columns), index=list(df.columns).index(default_col))

            if st.button('▶️ Run Batch Analysis'):
                texts = df[text_col].astype(str).fillna('')
                preds = predict_labels(texts.tolist())
                labels = [LABEL_MAP.get(int(p), str(p)) for p in preds]

                out_df = df.copy()
                out_df['Predicted Sentiment'] = labels

                # KPI row
                counts = pd.Series(labels).value_counts().reindex(['Positive', 'Negative']).fillna(0).astype(int)
                total = int(len(labels))
                c1, c2, c3 = st.columns(3)
                c1.metric('🧾 Total', f'{total}')
                c2.metric('😊 Positive', f"{counts.get('Positive', 0)}")
                c3.metric('😡 Negative', f"{counts.get('Negative', 0)}")

                st.subheader('🔎 Preview of Results')
                st.dataframe(out_df.head(20), use_container_width=True)

                # Pie chart of distribution
                st.subheader('🥧 Sentiment Distribution')
                fig, ax = plt.subplots(figsize=(2.8, 2.8))
                ax.pie(
                    [counts.get('Positive', 0), counts.get('Negative', 0)],
                    labels=['Positive', 'Negative'],
                    autopct='%1.1f%%', startangle=140,
                    colors=['#2ecc71', '#e74c3c']
                )
                ax.axis('equal')
                st.pyplot(fig, clear_figure=True)

                # Download button
                csv_buf = io.StringIO()
                out_df.to_csv(csv_buf, index=False)
                st.download_button(
                    label='⬇️ Download predictions as CSV',
                    data=csv_buf.getvalue(),
                    file_name='sentiment_predictions.csv',
                    mime='text/csv'
                )

                if hasattr(st, 'toast'):
                    st.toast(f'Finished predicting {total} reviews ✅', icon='🎉')

    else:
        st.info('📤 Awaiting CSV upload to run batch predictions.')

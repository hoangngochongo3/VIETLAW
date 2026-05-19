import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client

MODEL_ID     = "mainguyen9/vietlegal-harrier-0.6b"
INSTRUCTION  = "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "
SUPABASE_URL = "https://zzmqadwqrhrxiexuhevn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp6bXFhZHdxcmhyeGlleHVoZXZuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkxNDM0NzgsImV4cCI6MjA5NDcxOTQ3OH0.1dKLhvvcKyWRRe5Fu5pJfNqimb0j92FiCPaH20Quw5A"


@st.cache_resource(show_spinner="Dang tai model...")
def load_model():
    return SentenceTransformer(MODEL_ID, device="cpu")


@st.cache_resource(show_spinner="Ket noi Supabase...")
def load_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_doc_count(supabase):
    try:
        res = supabase.table("legal_passages").select("id", count="exact").execute()
        return res.count or 0
    except:
        return 0


def retrieve(query, model, supabase, top_k):
    q_emb = model.encode(
        [INSTRUCTION + query],
        normalize_embeddings=True
    )[0].tolist()
    res = supabase.rpc("match_passages", {
        "query_embedding": q_emb,
        "match_count": top_k
    }).execute()
    return res.data


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="VietLegal Search", page_icon="⚖️", layout="wide")
st.title("⚖️ VietLegal Harrier 0.6B")
st.caption(f"Model: `{MODEL_ID}` · Nguon: Zalo AI Legal Text Retrieval · CPU inference")

model     = load_model()
supabase  = load_supabase()
doc_count = get_doc_count(supabase)

with st.sidebar:
    st.header("Cai dat")
    top_k = st.slider("So ket qua (Top-K)", min_value=1, max_value=10, value=5)
    st.divider()
    st.metric("Van ban trong DB", f"{doc_count:,}")
    st.caption("Supabase: `vietlegal` · Singapore")
    st.caption("Dataset: GreenNode/zalo-ai-legal-text-retrieval-vn")

st.subheader("Nhap cau hoi phap ly")
query = st.text_input(
    label="Cau hoi",
    placeholder="VD: Thu tuc dang ky thanh lap doanh nghiep gom nhung buoc nao?",
    label_visibility="collapsed",
)

sample_queries = [
    "Thu tuc dang ky thanh lap doanh nghiep?",
    "Nguoi lao dong bi sa thai trai phap luat co quyen gi?",
    "Hop dong lao dong vo hieu trong truong hop nao?",
    "Tien luong lam them gio duoc tinh nhu the nao?",
    "Quyen cua nguoi tieu dung khi mua hang kem chat luong?",
]

st.caption("Cau hoi mau:")
cols = st.columns(len(sample_queries))
for col, sq in zip(cols, sample_queries):
    if col.button(sq, use_container_width=True):
        query = sq

if query:
    if doc_count == 0:
        st.warning("Chua co du lieu trong Supabase. Chay notebook Colab de nap data truoc.")
    else:
        with st.spinner("Dang tim kiem..."):
            results = retrieve(query, model, supabase, top_k)

        st.divider()
        st.subheader(f"Ket qua cho: *{query}*")

        if not results:
            st.error("Khong tim thay ket qua phu hop.")
        else:
            for rank, row in enumerate(results, 1):
                score   = float(row.get("similarity", 0))
                content = row.get("content", "")
                title   = row.get("title", "")
                doc_id  = row.get("doc_id", "")

                with st.container(border=True):
                    col1, col2 = st.columns([1, 11])
                    col1.markdown(f"### #{rank}")
                    with col2:
                        if title:
                            st.markdown(f"**{title}**")
                        if doc_id:
                            st.caption(f"doc_id: `{doc_id}`")
                        st.markdown(content[:800] + ("..." if len(content) > 800 else ""))
                    st.progress(
                        min(int(score * 100), 100),
                        text=f"Similarity: {score:.4f}"
                    )
else:
    st.info("Nhap cau hoi hoac chon cau hoi mau de bat dau tim kiem.")

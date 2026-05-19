import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client

MODEL_ID     = "mainguyen9/vietlegal-harrier-0.6b"
INSTRUCTION  = "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "
SUPABASE_URL = "https://zzmqadwqrhrxiexuhevn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp6bXFhZHdxcmhyeGlleHVoZXZuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkxNDM0NzgsImV4cCI6MjA5NDcxOTQ3OH0.1dKLhvvcKyWRRe5Fu5pJfNqimb0j92FiCPaH20Quw5A"


@st.cache_resource(show_spinner="Đang tải model...")
def load_model():
    return SentenceTransformer(MODEL_ID, device="cpu")


@st.cache_resource(show_spinner="Đang kết nối Supabase...")
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
st.caption(f"Model: `{MODEL_ID}` · Nguồn: Zalo AI Legal Text Retrieval · CPU inference")

model     = load_model()
supabase  = load_supabase()
doc_count = get_doc_count(supabase)

with st.sidebar:
    st.header("Cài đặt")
    top_k = st.slider("Số kết quả (Top-K)", min_value=1, max_value=10, value=5)
    st.divider()
    st.metric("Văn bản trong DB", f"{doc_count:,}")
    st.caption("Supabase: `vietlegal` · Singapore")
    st.caption("Dataset: GreenNode/zalo-ai-legal-text-retrieval-vn")

st.subheader("Nhập câu hỏi pháp lý")
query = st.text_input(
    label="Câu hỏi",
    placeholder="VD: Thủ tục đăng ký thành lập doanh nghiệp gồm những bước nào?",
    label_visibility="collapsed",
)

sample_queries = [
    "Thủ tục đăng ký thành lập doanh nghiệp?",
    "Người lao động bị sa thải trái pháp luật có quyền gì?",
    "Hợp đồng lao động vô hiệu trong trường hợp nào?",
    "Tiền lương làm thêm giờ được tính như thế nào?",
    "Quyền của người tiêu dùng khi mua hàng kém chất lượng?",
]

st.caption("Câu hỏi mẫu:")
cols = st.columns(len(sample_queries))
for col, sq in zip(cols, sample_queries):
    if col.button(sq, use_container_width=True):
        query = sq

if query:
    if doc_count == 0:
        st.warning("Chưa có dữ liệu trong Supabase. Chạy notebook Colab để nạp dữ liệu trước.")
    else:
        with st.spinner("Đang tìm kiếm..."):
            results = retrieve(query, model, supabase, top_k)

        st.divider()
        st.subheader(f"Kết quả cho: *{query}*")

        if not results:
            st.error("Không tìm thấy kết quả phù hợp.")
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
                        text=f"Độ tương đồng: {score:.4f}"
                    )
else:
    st.info("Nhập câu hỏi hoặc chọn câu hỏi mẫu để bắt đầu tìm kiếm.")

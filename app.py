import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_ID = "mainguyen9/vietlegal-harrier-0.6b"
INSTRUCTION = "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "

SAMPLE_PASSAGES = [
    "Điều 27. Trình tự, thủ tục đăng ký doanh nghiệp: Người thành lập doanh nghiệp nộp hồ sơ đăng ký doanh nghiệp tại Cơ quan đăng ký kinh doanh. Cơ quan đăng ký kinh doanh có trách nhiệm xem xét tính hợp lệ của hồ sơ và cấp Giấy chứng nhận đăng ký doanh nghiệp trong thời hạn 03 ngày làm việc.",
    "Điều 41. Nghĩa vụ của người sử dụng lao động khi đơn phương chấm dứt hợp đồng lao động trái pháp luật: Phải nhận người lao động trở lại làm việc theo hợp đồng lao động đã giao kết; phải trả tiền lương, đóng bảo hiểm xã hội trong những ngày người lao động không được làm việc và phải trả thêm cho người lao động một khoản tiền ít nhất bằng 02 tháng tiền lương.",
    "Điều 49. Hợp đồng lao động vô hiệu toàn bộ khi: toàn bộ nội dung của hợp đồng lao động vi phạm pháp luật; người giao kết hợp đồng lao động không đúng thẩm quyền hoặc vi phạm nguyên tắc giao kết hợp đồng lao động quy định tại Điều 15 của Bộ luật này.",
    "Điều 15. Vốn điều lệ của công ty trách nhiệm hữu hạn khi đăng ký thành lập doanh nghiệp là tổng giá trị phần vốn góp của các thành viên cam kết góp vào công ty. Thành viên phải góp vốn cho công ty đủ và đúng loại tài sản đã cam kết trong thời hạn 90 ngày kể từ ngày cấp Giấy chứng nhận đăng ký doanh nghiệp.",
    "Điều 117. Hợp đồng lao động phải được giao kết bằng văn bản và được làm thành 02 bản, người lao động giữ 01 bản, người sử dụng lao động giữ 01 bản, trừ trường hợp quy định tại khoản 2 Điều này.",
    "Điều 8. Quyền của người lao động: Làm việc; tự do lựa chọn việc làm, nơi làm việc, nghề nghiệp, học nghề, nâng cao trình độ nghề nghiệp; không bị phân biệt đối xử, cưỡng bức lao động, quấy rối tình dục tại nơi làm việc.",
    "Điều 134. Tiền lương làm thêm giờ: Người lao động làm thêm giờ được trả lương tính theo đơn giá tiền lương hoặc tiền lương thực trả theo công việc đang làm. Vào ngày thường ít nhất bằng 150%; vào ngày nghỉ hằng tuần ít nhất bằng 200%; vào ngày nghỉ lễ, tết ít nhất bằng 300%.",
    "Điều 206. Điều kiện thành lập trường đại học: Có đội ngũ giảng viên cơ hữu đảm bảo tỷ lệ sinh viên/giảng viên không quá 25 sinh viên/01 giảng viên; có cơ sở vật chất, thiết bị đáp ứng yêu cầu hoạt động của trường.",
]


@st.cache_resource(show_spinner="Đang tải model...")
def load_model():
    return SentenceTransformer(MODEL_ID, device="cpu")


@st.cache_data(show_spinner="Đang encode văn bản...")
def encode_passages(_model, passages: tuple):
    return _model.encode(list(passages), batch_size=4, normalize_embeddings=True)


def retrieve(query: str, model, passage_embs: np.ndarray, passages: list, top_k: int):
    full_query = INSTRUCTION + query
    q_emb = model.encode([full_query], normalize_embeddings=True)[0]
    scores = q_emb @ passage_embs.T
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(passages[i], float(scores[i])) for i in ranked]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="VietLegal Search", page_icon="⚖️", layout="wide")

st.title("⚖️ VietLegal Harrier 0.6B")
st.caption(f"Model: `{MODEL_ID}` · CPU inference · Semantic legal search")

with st.sidebar:
    st.header("Cài đặt")
    top_k = st.slider("Số kết quả trả về (Top-K)", min_value=1, max_value=5, value=3)
    st.caption(f"Đang dùng **{len(SAMPLE_PASSAGES)}** đoạn văn bản mẫu.")

passages = SAMPLE_PASSAGES
model = load_model()
passage_embs = encode_passages(model, tuple(passages))

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
]

st.caption("Câu hỏi mẫu:")
cols = st.columns(len(sample_queries))
for col, sq in zip(cols, sample_queries):
    if col.button(sq, use_container_width=True):
        query = sq

if query:
    with st.spinner("Đang tìm kiếm..."):
        results = retrieve(query, model, passage_embs, passages, top_k)

    st.divider()
    st.subheader(f"Kết quả cho: *{query}*")

    for rank, (passage, score) in enumerate(results, 1):
        color = "#1f77b4" if rank == 1 else "#666"
        bar_width = int(score * 100)
        with st.container(border=True):
            col1, col2 = st.columns([1, 11])
            col1.markdown(f"### #{rank}")
            col2.markdown(passage)
            st.progress(min(bar_width, 100), text=f"Score: {score:.4f}")
else:
    st.info("Nhập câu hỏi hoặc chọn câu hỏi mẫu để bắt đầu tìm kiếm.")

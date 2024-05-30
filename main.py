import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from pypdf import PdfReader
import os
import logging
import json
import base64

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

logging.basicConfig(level=logging.INFO)

title = "RAG"
Bot_img = "imgs/RAG_icon.jpg"
a_avatar_img = "imgs/assistant.png"
u_avatar_img = "imgs/user.png"
model_engine = "llama3-70b-8192"

st.set_page_config(
    page_title = f"{title} - An Intelligent Streamlit Assistant",
    page_icon = Bot_img,
    layout = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "Get help": "https://github.com/ここ",
        "Report a bug": "https://github.com/AdieLaine/ここ",
        "About": f"""
            ## {title}
            
            **GitHub**: https://github.com/ここ/
            
            チャットアプリです。追記予定。
        """
    }
)

st.title("RAG SAMPLE")

@st.cache_data(show_spinner=False)
def embeddings():
    model_path = f"intfloat/multilingual-e5-base"

    # ローカル版
    # model_path = f"/models/multilingual-e5-large"

    return HuggingFaceEmbeddings(model_name=model_path)

@st.cache_data(show_spinner=False)
def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

@st.cache_data(show_spinner=False)
def load_initial_values():
    path = "data/initial_values.json"
    return load_json(path)

@st.cache_data(show_spinner=False)
def get_avatar_image(role):
    if role == "assistant":
        return a_avatar_img
    elif role == "user":
        return u_avatar_img
    else:
        return None

@st.cache_data(show_spinner=False)
def process_pdf_files_and_create_faiss_index(files):
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="。",
        chunk_size=1000,
        chunk_overlap=15 # 15文字までは重複OK
    )

    docs = text_splitter.split_text(text)
    docs = [Document(page_content=doc) for doc in docs]
    db = FAISS.from_documents(docs, embeddings())

    return db

@st.cache_data(show_spinner=False)
def generate(messages, _source, stream=False):
    system_prompt = [
        {
            "role": "system",
            "content": f'''
    あなたは優秀な日本人のアシスタントです。
    以下の情報のみから、必ず日本語で回答を生成してください。
    情報が無い場合は、「情報ないっスけど、一般的には、」から回答を生成してください。

    情報１：{_source[0]}
    情報２：{_source[1]}
    情報３：{_source[2]}
    '''
        },
    ]

    completion = client.chat.completions.create(
        model=model_engine,
        messages=system_prompt + messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=stream,
        stop=None,
    )

    result = ""

    if stream == True:
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""
    else:
        result = completion.choices[0].message.content or ""

    return result

def create_slider(st):
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.system_prompt = []

    if not st.session_state.history:
        initial_values = load_initial_values()
        initial_bot_message = initial_values.get("initial_bot_message", "")
        prompt_text = initial_values.get("prompt_text", "")
        st.session_state.system_prompt = [
            {"role": "system", "content": prompt_text}
            ]
        st.session_state.history = [
            {"role": "assistant", "content": initial_bot_message}
            ]

    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow:
                0 0 5px #003300,
                0 0 10px #006600,
                0 0 15px #009900,
                0 0 20px #00CC00,
                0 0 25px #00FF00,
                0 0 30px #33FF33,
                0 0 35px #66FF66;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Function to convert image to base64
    def img_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Load and display sidebar image with glowing effect
    img_path = Bot_img
    img_base64 = img_to_base64(img_path)
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # Sidebar for Mode Selection
    mode = st.sidebar.radio("Select PDF:", options=["サンプル", "手動UP"], index=0)

    if "db" not in st.session_state:
        st.session_state.db = None

    if mode == "手動UP":
        # ファイルアップローダーの作成
        uploaded_files = st.sidebar.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)

        # アップロードされたファイルを処理
        if uploaded_files:
            st.session_state.db = process_pdf_files_and_create_faiss_index(uploaded_files)
    else:
        # サンプルPDFフォルダから読み取り
        # sample_pdf_folder = "samplePDF"
        # files = [os.path.join(sample_pdf_folder, file) for file in os.listdir(sample_pdf_folder) if file.endswith(".pdf")]
        # st.session_state.db = process_pdf_files_and_create_faiss_index(files)

        # ローカルから読込
        st.session_state.db = FAISS.load_local("./sample_db", embeddings(), allow_dangerous_deserialization=True)

    st.sidebar.markdown("---")

    show_basic_info = st.sidebar.toggle("Basic Interactions", value=True)
    if show_basic_info:
        st.sidebar.markdown("""
        ### Basic Interactions

        * **サンプル**: 以下の情報を基に回答を生成します。
        * **参照元**:
          + 「研究開発型スタートアップへのファンディングの在り方」に関する基本方針
          + AI に関する暫定的な論点整理
          + AI戦略2022
        """)
    st.sidebar.markdown("---")

def create_pdf_viewer(st):
    folder_path = "samplePDF"
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    pdf_file = st.selectbox("Select PDF file", pdf_files)

    if pdf_file:
        pdf_path = os.path.join(folder_path, pdf_file)
        with open(pdf_path, "rb") as f:
            binary_data = f.read()
        pdf_viewer(input=binary_data)

def main():
    create_slider(st)

    tabs = ["Chat", "PDF Viewer"]
    tab = st.tabs(tabs)

    with tab[1]:
        create_pdf_viewer(st)

    for message in st.session_state.history[-20:]:
        role = message["role"]
        with st.chat_message(role, avatar=get_avatar_image(role)):
            st.markdown(message["content"])

    chat_input = st.chat_input("メッセージを送信する:")
    if chat_input:
        st.session_state.history.append({"role": "user", "content": chat_input})
        with st.chat_message("user", avatar=get_avatar_image("user")):
            st.markdown(chat_input)

        with st.chat_message("assistant", avatar=get_avatar_image("assistant")):
            source = st.session_state.db.similarity_search(chat_input)
            result = generate(st.session_state.history[-20:], source, True)
            st.markdown(result)
            st.session_state.history.append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()
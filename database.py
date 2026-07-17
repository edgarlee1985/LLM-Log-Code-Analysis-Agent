import os
import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from tree_sitter import Parser, Query, QueryCursor
from tree_sitter import Language as TSLanguage
import tree_sitter_python as tspython
import tree_sitter_cpp as tscpp

# 定義想讀取的檔案副檔名
ALLOWED_EXTENSIONS = {'.h', '.cpp', '.ts', '.ui', '.css', '.txt', '.json'}
# 建立副檔名與 LangChain Language 的對應關係
EXTENSION_MAPPING = {
    '.cpp': Language.CPP, '.h': Language.CPP, '.hpp': Language.CPP, '.c': Language.C,
    '.py': Language.PYTHON, '.js': Language.JS, '.ts': Language.TS,
    '.html': Language.HTML, '.css': Language.HTML, '.md': Language.MARKDOWN,
}
# 定義要忽略的資料夾
IGNORE_DIRS = {'.git', '.vscode', 'build', 'venv', '.venv', 'dist'}


def get_ast_parser_and_query(ext: str):
    """根據副檔名回傳對應的 Tree-sitter Parser 與 Query 語法"""
    parser = Parser()
    query_str = ""
    ts_language = None # 改用 ts_language 變數名稱
    
    if ext in ['.py']:
        # 使用別名 TSLanguage 進行實例化
        ts_language = TSLanguage(tspython.language())
        parser.language = ts_language
        
        # 抓取 Python 的 class 與 function
        query_str = """
        (class_definition) @class
        (function_definition) @function
        """
    elif ext in ['.cpp', '.h', '.hpp', '.c']:
        # 使用別名 TSLanguage 進行實例化
        ts_language = TSLanguage(tscpp.language())
        parser.language = ts_language
        
        # 抓取 C/C++ 的 class、struct 與 function
        query_str = """
        (class_specifier) @class
        (struct_specifier) @struct
        (function_definition) @function
        """
    else:
        return None, None, None
        
    return parser, ts_language, query_str

def ast_chunk_document(doc: Document) -> list[Document]:
    """將單一檔案原始碼轉化為基於 AST 的多個區塊 (僅支援 tree-sitter >= 0.22.0)"""
    source_path = doc.metadata.get("source", "")
    ext = os.path.splitext(source_path)[1].lower()
    
    parser, ts_language, query_str = get_ast_parser_and_query(ext)
    
    if not parser or not ts_language:
        return []

    source_bytes = doc.page_content.encode('utf-8')
    tree = parser.parse(source_bytes)
    
    query = Query(ts_language, query_str)
    cursor = QueryCursor(query)
    
    # 新版 captures 直接回傳 dict: { "capture_name": [node1, node2] }
    raw_captures = cursor.captures(tree.root_node)
    
    ast_docs = []
    normalized_captures = []
    
    # 攤平 dict 並重新包裝成 (node, capture_name)
    for capture_name, nodes in raw_captures.items():
        for node in nodes:
            normalized_captures.append((node, capture_name))
            
    # 依照 AST 節點在原始碼中的起始位置排序，確保程式碼順序正確
    normalized_captures.sort(key=lambda x: x[0].start_byte)
        
    for node, capture_name in normalized_captures:
        snippet = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
        
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        metadata = {
            "source": source_path,
            "start_line": start_line,
            "end_line": end_line,
            "ast_type": capture_name
        }
        
        numbered_snippet = "\n".join(
            [f"{start_line + i} | {line}" for i, line in enumerate(snippet.split('\n'))]
        )
        
        ast_docs.append(Document(page_content=numbered_snippet, metadata=metadata))
        
    return ast_docs


def get_files_from_repo(repo_path):
    """走訪資料夾，取得所有符合條件的檔案路徑"""
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        # 移除不需要掃描的資料夾
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ALLOWED_EXTENSIONS:
                file_paths.append(os.path.join(root, file))
    return file_paths

def gen_faii_index_from_path(repo_path: str, db_dir: str, embeddings):
    print(f"開始掃描資料夾: {repo_path}")
    file_paths = get_files_from_repo(repo_path)
    
    # 讀取檔案內容並建立 Document 物件
    documents = []
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                doc = Document(page_content=content, metadata={"source": path})
                documents.append(doc)
        except Exception as e:
            print(f"讀取檔案失敗 {path}: {e}")

    print(f"共讀取了 {len(documents)} 個檔案。開始動態切塊...")

    # 動態切割文本 (Dynamic Text Splitting)
    all_chunks = []
    
    # 建立一個通用的 Fallback/二次切塊 Splitter
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200
    )
    
    for doc in documents:
        # 1. 嘗試使用 AST 切塊
        ast_chunks = ast_chunk_document(doc)
        
        if ast_chunks:
            # 2. 檢查 AST 切出來的區塊是否過大，若過大則進行二次切割
            for ast_chunk in ast_chunks:
                if len(ast_chunk.page_content) > 1500:
                    sub_chunks = fallback_splitter.split_documents([ast_chunk])
                    all_chunks.extend(sub_chunks)
                else:
                    all_chunks.append(ast_chunk)
        else:
            # 如果是前端檔案 (如 .html, .css) 或未支援 AST 的語言，退回你原本的邏輯
            source_path = doc.metadata.get("source", "")
            ext = os.path.splitext(source_path)[1].lower()
            
            if ext in EXTENSION_MAPPING:
                lang = EXTENSION_MAPPING[ext]
                lang_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang, chunk_size=1000, chunk_overlap=200
                )
                all_chunks.extend(lang_splitter.split_documents([doc]))
            else:
                all_chunks.extend(fallback_splitter.split_documents([doc]))

    print(f"所有檔案已切割成 {len(all_chunks)} 個區塊 (Chunks)。")
    
    # 建立 FAISS 向量資料庫
    print("正在建立 FAISS 向量資料庫 (這可能需要幾分鐘的時間)...")
    vector_db = FAISS.from_documents(all_chunks, embeddings)

    # 儲存資料庫到本地端
    vector_db.save_local(db_dir)
    print(f"完成！向量資料庫已儲存至: {db_dir}")

    # ================= 新增區塊：建立並儲存 BM25 =================
    print("正在建立 BM25 關鍵字檢索器...")
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 3
    
    bm25_path = os.path.join(db_dir, "bm25_retriever.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"完成！BM25 檢索器已儲存至: {bm25_path}")


def build_or_load_retriever(repo_dir: str, db_dir: str, embeddings) -> EnsembleRetriever:
    """初始化並回傳混合檢索器 (EnsembleRetriever)"""
    bm25_path = os.path.join(db_dir, "bm25_retriever.pkl")
    faiss_index_path = os.path.join(db_dir, "index.faiss")
    
    if not (os.path.isdir(db_dir) and os.path.exists(bm25_path) and os.path.exists(faiss_index_path)):
        gen_faii_index_from_path(repo_dir, db_dir, embeddings)

    vector_db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

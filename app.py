import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from pathlib import Path

# 現代 LangChain 工具與 LCEL
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
import pickle
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from langchain_google_genai import ChatGoogleGenerativeAI

import subprocess

BUG_REPORT_DIR = "Report Folder" # Bug Report 路徑
REPO_PATH = "Git Folder"  # Git repo 路徑
FAISS_DB_DIR = "./faiss_index"        # 向量資料庫儲存的資料夾名稱

MODEL_NAME = "qwen3"

EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# 定義你想讀取的檔案副檔名
ALLOWED_EXTENSIONS = {'.h', '.cpp', '.ts', '.ui', '.css', '.txt', '.json'}
# 建立副檔名與 LangChain Language 的對應關係
EXTENSION_MAPPING = {
    # C / C++ 家族
    '.cpp': Language.CPP,
    '.h': Language.CPP,
    '.hpp': Language.CPP,
    '.c': Language.C,
    
    # 其他常見程式語言
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.ts': Language.TS,  # TypeScript (若是 Qt 的語系檔則退回通用切塊即可)
    '.html': Language.HTML,
    '.css': Language.HTML, # CSS 結構與 HTML 類似，通常用 HTML 也可以稍微保持結構，或用通用
    '.md': Language.MARKDOWN,
}
# 定義要忽略的資料夾
IGNORE_DIRS = {'.git', '.vscode', 'build', 'venv', '.venv', 'dist'}

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

def gen_faii_index_from_path(repo_path):
    print(f"開始掃描資料夾: {repo_path}")
    file_paths = get_files_from_repo(repo_path)
    
    # 1. 讀取檔案內容並建立 Document 物件
    documents = []
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                # 將每一行加上行號，例如： "12 | void doSomething() {"
                numbered_content = "".join([f"{i+1} | {line}" for i, line in enumerate(lines)])
                
                # 用帶有行號的字串去建立 Document
                doc = Document(page_content=numbered_content, metadata={"source": path})

                #content = f.read()
                #doc = Document(page_content=content, metadata={"source": path})
                documents.append(doc)
        except Exception as e:
            print(f"讀取檔案失敗 {path}: {e}")

    print(f"共讀取了 {len(documents)} 個檔案。開始動態切塊...")

    # 2. 動態切割文本 (Dynamic Text Splitting)
    all_chunks = []
    
    for doc in documents:
        # 從 metadata 取得原始路徑，並萃取副檔名並轉小寫
        source_path = doc.metadata.get("source", "")
        ext = os.path.splitext(source_path)[1].lower()
        
        # 判斷是否有對應的專屬語言 Splitter
        if ext in EXTENSION_MAPPING:
            lang = EXTENSION_MAPPING[ext]
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=1000,
                chunk_overlap=200
            )
        else:
            # 找不到對應語言，退回使用通用字元切塊
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
        # 對這個單一檔案進行切塊，並加入總清單中
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    print(f"所有檔案已切割成 {len(all_chunks)} 個區塊 (Chunks)。")
    
    # 3. 初始化嵌入模型 (Embedding Model)
    # 這裡使用 HuggingFace 開源且輕量的模型，適合一般文本與程式碼
    print("正在下載/載入 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDINGS_MODEL_NAME)
    
    # 4. 建立 FAISS 向量資料庫
    print("正在建立 FAISS 向量資料庫 (這可能需要幾分鐘的時間)...")
    vector_db = FAISS.from_documents(all_chunks, embeddings)

    # 5. 儲存資料庫到本地端
    vector_db.save_local(FAISS_DB_DIR)
    print(f"完成！向量資料庫已儲存至: {FAISS_DB_DIR}")

    # ================= 新增區塊：建立並儲存 BM25 =================
    print("正在建立 BM25 關鍵字檢索器...")
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 3
    
    bm25_path = os.path.join(FAISS_DB_DIR, "bm25_retriever.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"完成！BM25 檢索器已儲存至: {bm25_path}")

# ==========================================
# 基礎設定 (使用本機 Ollama 舉例)
# ==========================================
llm = ChatOllama(model=MODEL_NAME, temperature=0, seed=50) # 也可以替換成 Llama-3.1 或 OpenAI

# 必須使用與建立時相同的 Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name = EMBEDDINGS_MODEL_NAME)

print(f"檢查 {FAISS_DB_DIR} 是否存在")
bm25_path = os.path.join(FAISS_DB_DIR, "bm25_retriever.pkl")
faiss_index_path = os.path.join(FAISS_DB_DIR, "index.faiss")
if not (os.path.isdir(FAISS_DB_DIR) and os.path.exists(bm25_path) and os.path.exists(faiss_index_path)):
    gen_faii_index_from_path(REPO_PATH)

# 載入本地端的 FAISS 資料庫
# allow_dangerous_deserialization=True 是因為 FAISS 使用 pickle，載入信任的本地檔案時需開啟此設定
vector_db = FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)

# 將 FAISS 轉為標準 Retriever
faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# ================= 新增區塊：載入 BM25 並組合 =================
bm25_path = os.path.join(FAISS_DB_DIR, "bm25_retriever.pkl")
with open(bm25_path, "rb") as f:
    bm25_retriever = pickle.load(f)

# 建立混合檢索器 (權重可依據測試結果微調，這裡先設定各 50%)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)


class SingleBugReport(BaseModel):
    """單一 Bug 的資料結構"""
    bug_id: str # 例如: "bug_000001"
    steps_to_reproduce: str # bug_xxxxxx.txt 內的文字內容
    logs: Dict[str, str] # 紀錄該 bug 底下所有的 log，格式為 { "application.log_1": "log內容..." }

def read_bug_report(report_dir: str) -> List[SingleBugReport]:
    """
    從指定目錄讀取所有的 Bug reports 並轉換為 SingleBugReport 物件列表。
    """
    base_path = Path(report_dir)
    bug_reports: List[SingleBugReport] = []

    # 檢查目標資料夾是否存在
    if not base_path.exists() or not base_path.is_dir():
        print(f"警告: 找不到目錄 {report_dir}")
        return bug_reports

    # 遍歷 base_path 下的所有項目
    for bug_dir in base_path.iterdir():
        # 確認該項目是資料夾，且名稱符合 "bug_" 開頭的格式
        if bug_dir.is_dir() and bug_dir.name.startswith("bug_"):
            bug_id = bug_dir.name
            steps_to_reproduce = ""
            logs: Dict[str, str] = {}
            
            # 1. 讀取操作步驟 txt 檔 (預期檔名為 bug_id.txt)
            txt_file_path = bug_dir / f"{bug_id}.txt"
            if txt_file_path.exists() and txt_file_path.is_file():
                # 建議加上 encoding="utf-8" 避免跨平台中文編碼錯誤
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    steps_to_reproduce = f.read()
            else:
                print(f"警告: {bug_id} 目錄下找不到 {bug_id}.txt")
            
            # 2. 讀取目錄下的所有 application.log_x 檔案
            for file_path in bug_dir.iterdir():
                if file_path.is_file() and file_path.name.startswith("application.log"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        logs[file_path.name] = f.read()
            
            # 將收集到的資料建立為 SingleBugReport 模型並加入列表
            report = SingleBugReport(
                bug_id=bug_id,
                steps_to_reproduce=steps_to_reproduce,
                logs=logs
            )
            bug_reports.append(report)

    # 可依據 bug_id 進行排序，確保回傳結果的一致性
    bug_reports.sort(key=lambda x: x.bug_id)
    
    return bug_reports

# ==========================================
# 步驟一：從 Log 中萃取「精準線索」（Log Parsing）
# ==========================================
# 1. 定義預期的 Log 結構
class LogClues(BaseModel):
    error_type: Optional[str] = Field(description="明確的錯誤類型，如 TypeError，若無則留空")
    file_name: Optional[str] = Field(description="Log 中提到的可能發生錯誤的檔案名稱")
    line_number: Optional[int] = Field(description="錯誤發生的行號，若無則為 -1", default=-1)
    semantic_issue: str = Field(description="將 Log 的行為總結為一句語意描述，例如：'事件迴圈重複觸發'")

def parse_report(bug_report: SingleBugReport) -> LogClues:
    """使用 LLM 將雜亂的 Log 轉換為結構化線索"""
    
    # 由於 bug_report.logs 是 Dict[str, str]，我們將其組合成單一字串
    # 加上檔名標籤 (例如 === application.log_1 ===)，讓 LLM 能區分不同的 log 來源
    combined_logs = "\n\n".join(
        f"=== {filename} ===\n{content}" 
        for filename, content in bug_report.logs.items()
    )
    
    # 如果 logs 是空的，給予預設提示避免 LLM 混淆
    if not combined_logs.strip():
        combined_logs = "此 Bug 沒有任何對應的 Log 紀錄。"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個 Log 分析系統。請從以下 Log 中萃取出關鍵資訊。如果沒有明確的 Exception，請推敲其語意異常行為。"),
        ("human", "{raw_log}")
    ])
    
    # 結合 LCEL 與 Structured Output 確保回傳 Pydantic 格式
    parser_chain = prompt | llm.with_structured_output(LogClues)
    
    # 將組合好的 logs 字串傳遞給 prompt 中的 {raw_log} 變數
    return parser_chain.invoke({"raw_log": combined_logs})


# ==========================================
# 步驟二：建立程式碼的檢索機制（Codebase RAG - Tools）
# ==========================================
# 定義給 Agent 使用的工具 (Tools)

@tool
def semantic_code_search(query: str) -> str:
    """當不知道具體檔名，但知道邏輯異常時使用。根據語意搜尋 Codebase。"""
    docs = ensemble_retriever.invoke(query)
    result = []
    for d in docs:
        source = d.metadata.get('source', 'Unknown')
        content = d.page_content
        result.append(f"【檔案: {source}】\n片段內容:\n{content}\n")
    return "\n\n---\n\n".join(result)

@tool
def read_code_snippet(file_name: str, start_line: int, end_line: int) -> str:
    """
    當已知確切檔案名稱與行號時，讀取該檔案特定範圍的程式碼。
    注意：file_name 必須是絕對路徑，不可以只有純檔名。
    """
    print(f"read_code_snippet, file_name = {file_name}, start_line = {start_line}, end_line = {end_line}")

    # 防呆檢查：如果檔案不存在，給 LLM 一個有建設性的錯誤提示
    if not os.path.exists(file_name):
        return (f"讀取檔案失敗: 找不到檔案 '{file_name}'。 請提供完整的檔案路徑後重試。")

    try:
        # ✅ 明確指定 UTF-8 編碼，並忽略/替換無法解析的字元
        with open(file_name, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()[start_line-1:end_line]
        return "".join(lines)
    except Exception as e:
        return f"讀取檔案失敗: {e}"

@tool
def get_git_blame(file_name: str, line_number: int) -> str:
    """
    查詢某行程式碼的 Git Blame，了解是誰在什麼時候修改了這行邏輯
    注意：file_name 必須是絕對路徑，不可以只有純檔名。
    """
    
    print(f"get_git_blame, file_name = {file_name}, line_number = {line_number}")

    # 1. 取得檔案所在的目錄
    work_dir = os.path.dirname(file_name)
    
    # 2. 取得純檔名 (例如：main.cpp)
    base_name = os.path.basename(file_name)
    
    # 3. 透過 cwd 參數指定在該檔案的目錄下執行指令
    try:
        return subprocess.check_output(
            ["git", "blame", "-L", f"{line_number},{line_number}", base_name],
            cwd=work_dir,
            text=True,
            stderr=subprocess.STDOUT # ✅ 將 Git 的錯誤訊息合併到輸出中捕獲
        )
    except subprocess.CalledProcessError as e:
        # 如果 LLM 傳了超過檔案行數的數字，明確告訴它錯在哪裡
        if "has only" in e.output:
            return f"Git Blame 失敗: 檔案沒有第 {line_number} 行，請重新確認該檔案的總行數。"
        return f"Git Blame 失敗: {e.output}"

@tool
def exact_keyword_search(keyword: str, file_extension: str = "") -> str:
    """
    當已知明確的變數名稱、函數名稱或錯誤訊息關鍵字時使用。
    進行整個 Codebase 的精確字串比對（類似 grep）。
    可以選擇性提供副檔名過濾 (例如: '.cpp' 或 '.py')。
    """
    print(f"exact_keyword_search, keyword = {keyword}, file_extension = {file_extension}")

    try:
        # 使用 git grep 是最快搜尋 repo 的方式 (假設 REPO_PATH 是一個 git repo)
        cmd = ["git", "grep", "-n", keyword]
        if file_extension:
            cmd.append(f"*{file_extension}")
            
        result = subprocess.check_output(
            cmd, 
            cwd=REPO_PATH, 
            text=True, 
            encoding='utf-8',    # 👈 強制要求以 UTF-8 解碼
            errors='replace',    # 👈 遇到無法解碼的亂碼時，替換成  而非崩潰
            stderr=subprocess.STDOUT
        )
        
        # 限制回傳長度避免 Context Window 爆掉
        lines = result.split('\n')
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... (還有 {len(lines) - 50} 筆結果，請提供更精確的關鍵字)"
        return result
    except subprocess.CalledProcessError:
        return f"找不到包含精確關鍵字 '{keyword}' 的程式碼。"

tools = [semantic_code_search, read_code_snippet, get_git_blame, exact_keyword_search]


# ==========================================
# 步驟三：導入 Agentic 迭代排查（Agentic Workflow）
# ==========================================
def run_debugging_agent(bug_report: SingleBugReport, log_clues: LogClues):
    """建立並執行負責除錯的 AgentRunner"""
    
    # 將原始 log 組合起來
    raw_logs = "\n\n".join([f"=== {k} ===\n{v}" for k, v in bug_report.logs.items()])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個資深的軟體工程師，任務是找出系統 Bug 的根本原因 (Root Cause)。
        
        【🚨 核心守則】
        1. 你的唯一目標是解決使用者提供的「當前 Bug 報告」，請勿發散去列出或修復 Codebase 中其他無關的問題。
        2. 請交叉比對「Bug 重現步驟」、「原始 Log」與「工具查詢回來的程式碼」，只有在邏輯鏈條完全吻合時，才判定為 Root Cause。
        
        【🔍 排查步驟】
        1. 先根據提供的 Log 解析線索，決定要使用哪個工具。
        2. 如果有明確檔名與行號，請先用 `read_code_snippet` 讀取該處上下文 (建議讀取錯誤行號前後 15 行)。
        3. 如果只有語意異常，請使用 `semantic_code_search` 找尋相關邏輯。
        4. 迭代檢查程式碼，直到你確信找到導致「此特定 Bug」的根本原因為止。
        5. 最後輸出一份詳細的 Bug 報告與修復建議。

         """),
        
        ("human", """請協助排查以下 Bug：
        
        【Bug ID】
        {bug_id}
        
        【重現步驟】
        {steps}
        
        【原始 Log】
        {logs}
        
        【初步解析線索】
        {log_clues}
        """),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_runner = AgentExecutor(agent=agent,
                                 tools=tools,
                                 verbose=True,
                                 max_iterations=15,
                                 early_stopping_method="generate"
                                 )
    
    # 傳入完整的上下文
    result = agent_runner.invoke({
        "bug_id": bug_report.bug_id,
        "steps": bug_report.steps_to_reproduce if bug_report.steps_to_reproduce else "無提供",
        "logs": raw_logs if raw_logs else "無提供",
        "log_clues": log_clues.model_dump_json(indent=2)
    })
    
    return result["output"]


# ==========================================
# 主程式：完整工作流整合
# ==========================================
if __name__ == "__main__":
    bug_reports = read_bug_report(BUG_REPORT_DIR)

    for report in bug_reports:
        print(f"開始分析 bug_id: {report.bug_id}")
        print("--- 階段一：啟動 Log 解析 ---")
        clues = parse_report(report)
        print(f"萃取線索: {clues}\n")
        
        print("--- 階段二 & 三：啟動 Agentic 迭代排查 ---")
        final_report = run_debugging_agent(report, clues)
        print("\n--- 最終 Bug 報告 ---")
        print(final_report)
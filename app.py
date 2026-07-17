import os
import configparser
from typing import List, Dict, Optional
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from pathlib import Path

import operator

# 現代 LangChain 工具與 LCEL
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import FAISS
import pickle
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from tree_sitter import Parser, Query, QueryCursor
from tree_sitter import Language as TSLanguage
import tree_sitter_python as tspython
import tree_sitter_cpp as tscpp

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from langchain_google_genai import ChatGoogleGenerativeAI

from tools import create_agent_tools

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

# 創建 configparser 物件
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

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

def gen_faii_index_from_path(repo_path):
    print(f"開始掃描資料夾: {repo_path}")
    file_paths = get_files_from_repo(repo_path)
    
    # 1. 讀取檔案內容並建立 Document 物件
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

    # 2. 動態切割文本 (Dynamic Text Splitting)
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
    
    # 3. 初始化嵌入模型 (Embedding Model)
    # 這裡使用 HuggingFace 開源且輕量的模型，適合一般文本與程式碼
    print("正在下載/載入 Embedding 模型...")
    embeddings = OllamaEmbeddings(model = config["Default"]["EmbeddingModelName"])
    
    # 4. 建立 FAISS 向量資料庫
    print("正在建立 FAISS 向量資料庫 (這可能需要幾分鐘的時間)...")
    vector_db = FAISS.from_documents(all_chunks, embeddings)

    # 5. 儲存資料庫到本地端
    vector_db.save_local(config["Default"]["FAISSDBDir"])
    print(f"完成！向量資料庫已儲存至: {config["Default"]["FAISSDBDir"]}")

    # ================= 新增區塊：建立並儲存 BM25 =================
    print("正在建立 BM25 關鍵字檢索器...")
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 3
    
    bm25_path = os.path.join(config["Default"]["FAISSDBDir"], "bm25_retriever.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"完成！BM25 檢索器已儲存至: {bm25_path}")


def build_or_load_retriever(repo_dir: str, db_dir: str, embeddings) -> EnsembleRetriever:
    """初始化並回傳混合檢索器 (EnsembleRetriever)"""
    bm25_path = os.path.join(db_dir, "bm25_retriever.pkl")
    faiss_index_path = os.path.join(db_dir, "index.faiss")
    
    if not (os.path.isdir(db_dir) and os.path.exists(bm25_path) and os.path.exists(faiss_index_path)):
        gen_faii_index_from_path(repo_dir)

    vector_db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

# ==========================================
# 基礎設定 (使用本機 Ollama 舉例)
# ==========================================
llm = ChatOllama(model=config["Default"]["ModelName"], temperature=0.0, seed=50, repeat_penalty=1.2, num_ctx=8192, num_predict=4096) # 也可以替換成 Llama-3.1 或 OpenAI

# 必須使用與建立時相同的 Embedding 模型
embeddings = OllamaEmbeddings(model = config["Default"]["EmbeddingModelName"])

class SingleBugReport(BaseModel):
    """單一 Bug 的資料結構"""
    bug_id: str = Field(default="", description="Bug 的唯一識別碼，例如: 'bug_000001'")
    steps_to_reproduce: str = Field(default="", description="記錄重現步驟，通常為 bug_xxxxxx.txt 內的文字內容")
    logs: Dict[str, str] = Field(default_factory=dict, description="紀錄該 bug 底下所有的 log，格式為 { 'application.log_1': 'log內容...' }")

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
    error_type: Optional[str] = Field(default=None, description="明確的錯誤類型，如 TypeError，若無則留空")
    file_name: Optional[str] = Field(
        default=None,
        description="Log 中提到的錯誤檔案路徑（請直接擷取 Log 中顯示的路徑，例如 /path/code/.../main.cpp 或 src/main.cpp，能抓多完整就抓多完整）"
    )
    line_number: Optional[int] = Field(default=-1, description="錯誤發生的行號，若無則為 -1")
    semantic_issue: str = Field(default="", description="將 Log 的行為總結為一句語意描述，例如：'事件迴圈重複觸發'")

def parse_report(bug_report: SingleBugReport) -> LogClues:
    """使用 LLM 將雜亂的 Log 轉換為結構化線索"""
    
    # 由於 bug_report.logs 是 Dict[str, str]，我們將其組合成單一字串
    # 加上檔名標籤 (例如 === application.log_1 ===)，讓 LLM 能區分不同的 log 來源
    combined_logs = "\n\n".join(
        f"=== {filename} ===\n{content}" 
        for filename, content in bug_report.logs.items()
    )

    # 將 Windows 路徑的反斜線替換為正斜線，避免 JSON 跳脫字元崩潰
    combined_logs = combined_logs.replace("\\", "/")
    
    # 如果 logs 是空的，給予預設提示避免 LLM 混淆
    if not combined_logs.strip():
        combined_logs = "此 Bug 沒有任何對應的 Log 紀錄。"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個專為「AI 檢索探員 (RAG Agent)」提供精確搜索彈藥的【資深技術日誌萃取引擎】。
你收到的原始資料可能混雜了「QA/RD 的口語對話」、「測試步驟」以及「系統崩潰日誌 (Log/Stack Trace)」。

【你的核心任務】
如同外科手術般，過濾掉所有人類的日常用語與情緒字眼，只留下能夠用來進行「精確字串搜尋 (grep)」或「程式碼語意檢索」的技術實體 (Technical Entities)。

【資料萃取嚴格守則】
1. 錯誤類型 (error_type):
   - 僅抓取標準的 Exception 命名或系統錯誤碼 (例如: NullReferenceException, TypeError, SIGSEGV, HTTP 500)。
   - 若只是邏輯錯誤 (如: 算錯錢、畫面卡住)，請留空 (null)。

2. 檔案與行號 (file_name & line_number):
   - 優先從 Stack Trace 或 Error Log 中尋找確實崩潰的檔案路徑。
   - 盡可能保留最完整的相對/絕對路徑 (如 src/controllers/user_controller.cpp)，不要只留純檔名。

3. 語意描述 (semantic_issue) 撰寫規範 (🚨 極度重要):
   - 你的輸出將直接作為向量搜尋庫 (FAISS) 的 Query，必須高度技術化。
   - 強制保留原始文本中的所有英文實體：包含變數名稱 (Variable)、類別 (Class)、函數 (Function / CamelCase / snake_case) 或 API 路由 (/api/v1/login)。
   - 將人類的「行為描述」轉換為「系統狀態異常描述」。
   - ❌ 錯誤示範 (太口語)："QA 說按下結帳按鈕後畫面卡住了，RD 覺得可能是 API 沒回傳，叫我看一下 login log。"
   - ✅ 正確示範 (具檢索價值)："觸發 Checkout 按鈕後發生 timeout，疑似 PaymentGateway 模組中的 fetch_user_token 未正確處理空值回傳。"

請保持冷靜、客觀，忽略無關痛癢的對話，輸出最精煉的技術線索。"""),
        ("human", "【原始 Bug 報告與 Log 紀錄】\n{raw_log}")
    ])
    
    # 結合 LCEL 與 Structured Output 確保回傳 Pydantic 格式
    parser_chain = prompt | llm.with_structured_output(LogClues)
    
    # 將組合好的 logs 字串傳遞給 prompt 中的 {raw_log} 變數
    return parser_chain.invoke({"raw_log": combined_logs})


# ==========================================
# 步驟二：建立程式碼的檢索機制（Codebase RAG - Tools）
# ==========================================



# ==========================================
# 步驟三：導入 Agentic 迭代排查（Agentic Workflow）
# ==========================================

# ==========================================
# 結構化資料模型 (Pydantic Models)
# ==========================================
class DetectiveCommand(BaseModel):
    """工程師開給探員的情報需求單"""
    hypothesis: str = Field(default="", description="目前正在驗證的假設，例如：'我懷疑 user_login 函數沒有正確處理 null 值'")
    action_type: str = Field(default="", description="要執行的檢索類型：READ_FILE, SEMANTIC_SEARCH, EXACT_KEYWORD 等")
    target_value: str = Field(default="", description="對應的關鍵字、檔案路徑或語意描述")
    focus_point: str = Field(default="", description="請探員特別注意什麼？例如：'請幫我確認這個 class 有沒有繼承 BaseUser'")

class EngineerEvaluation(BaseModel):
    """工程師的深度分析與決策報告"""
    step_by_step_reasoning: str = Field(default="", description="請詳細推演你的邏輯鏈條：上一步看到了什麼？符合預期嗎？接下來要驗證什麼？")
    is_resolved: bool = Field(default=False, description="是否已經找到 Root Cause 並能提出修復方案？")
    next_search_request: Optional[DetectiveCommand] = Field(default=None, description="如果尚未解決，指派給探員的下一步檢索指令")
    final_report: Optional[str] = Field(default=None, description="如果 is_resolved 為 True，輸出完整修復報告；否則留空")

# ==========================================
# 全局狀態 (GraphState)
# ==========================================

class GraphState(TypedDict):
    bug_id: str
    steps: str
    logs: str
    log_clues: str
    
    # 使用 operator.add 讓每次的調查紀錄自動附加，形成對話歷史
    investigation_history: Annotated[List[str], operator.add] 
    
    # 存放 Engineer 開出的具體指令 (DetectiveCommand 的 dict 格式)
    current_request: Optional[dict] 
    
    iterations: int
    is_resolved: bool
    final_report: str

# ==========================================
# 3. Prompts (工程師的 System & Human Prompt)
# ==========================================

engineer_system_prompt = """你是一位頂尖的資深軟體工程師，負責帶領團隊進行複雜系統的除錯 (Debugging)。
你的唯一目標是找出系統 Bug 的根本原因 (Root Cause) 並提出精確的修復方案。

你目前正在與一位「檢索探員 (Detective)」合作。探員負責深入 Codebase 撈取程式碼，而你負責指揮他。

【💡 工作模式：假設驅動 (Hypothesis-Driven)】
你必須嚴格遵循以下思考循環：
1. 觀察 (Observe)：仔細閱讀使用者的 Bug 重現步驟與原始系統 Log。
2. 回顧 (Review)：檢視你與探員的「歷史調查紀錄」。特別留意【執行失敗 / 查無資料】的回報，這代表你先前的假設路徑或關鍵字錯誤，你必須改變策略，絕對不要重複發送相同的無效指令。
3. 推理 (Reasoning)：將 Log 線索與探員帶回的程式碼進行交叉比對，找出邏輯斷層。
4. 假設 (Hypothesis)：針對可能出錯的邏輯提出具體假設（例如："我懷疑 `calculate_total` 沒有處理空陣列"）。
5. 行動 (Action)：如果你確信已找到 Root Cause，宣告結案並撰寫報告；若證據不足，開立精確的情報需求單給探員。

【🚨 核心守則】
1. 絕不憑空捏造：絕對不要幻想或猜測未被檢索出來的程式碼邏輯。如果你沒親眼看到那段程式碼，就請探員去讀取。
2. 保持專注：只針對導致「當前 Bug Log」的程式碼進行排查，不要發散去檢查無關的模組或提出無謂的重構建議。
3. 善用語意搜尋：如果你不知道具體的檔名或函數名稱，不要瞎猜路徑，請指示探員使用語意搜尋 (SEMANTIC_SEARCH) 來尋找相關邏輯。

【📝 輸出要求】
你必須嚴格遵守 JSON 格式輸出。請根據你是否已經找到 Root Cause，選擇以下兩種 JSON 格式之一進行輸出：

情況 A：尚未解決，需要探員繼續檢索
{{
    "step_by_step_reasoning": "我剛剛看到...這代表...我接下來需要確認...",
    "is_resolved": false,
    "next_search_request": {{
        "hypothesis": "我懷疑 Button 沒有綁定正確的 index",
        "action_type": "READ_FILE",
        "target_value": "/path/code/src/GenTestLogProject.cpp",
        "focus_point": "檢查 OnButtonClikced 的實作邏輯"
    }},
    "final_report": ""
}}

情況 B：已經找到 Root Cause，準備結案
{{
    "step_by_step_reasoning": "根據探員的回報，我發現按鈕 0 的程式碼漏了...",
    "is_resolved": true,
    "next_search_request": null,
    "final_report": "Root Cause: XXX。 修復建議: 將 YYY 改為 ZZZ。"
}}
"""

engineer_human_prompt = """
【Bug 重現步驟】
{steps}

【系統 Log 原始紀錄】
{logs}

【歷史調查紀錄 (Investigation History)】
{investigation_history}

====================================
請基於以上資訊，進行深度思考並給出你的評估：
1. 如果你需要更多線索，請填寫 next_search_request 派發任務給探員。
2. 如果你已經確定根本原因，請將 is_resolved 設為 true，並給出 final_report。
"""


# ==========================================
# 4. 核心節點 (Nodes)
# ==========================================

def engineer_node(state: GraphState):
    print(f"\n[Engineer Node] 開始深度分析 (第 {state['iterations']} 次迭代)...")
    
    # 將歷史紀錄從 List[str] 組裝成單一字串，讓 LLM 閱讀
    if not state.get("investigation_history"):
        history_text = "目前尚無調查紀錄，這是第一次推論。請根據 Log 提出第一個假設並指派探員去檢索。"
    else:
        history_text = "\n\n".join(state["investigation_history"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", engineer_system_prompt),
        ("human", engineer_human_prompt)
    ])

    structured_llm = llm.with_structured_output(EngineerEvaluation)
    
    # 建立管線
    analysis_chain = (
        prompt 
        | structured_llm 
    ).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True
    )
    
    # 執行，並傳入格式說明給 LLM
    evaluation = analysis_chain.invoke({
        "steps": state["steps"],
        "logs": state["logs"],
        "investigation_history": history_text
    })
    
    print(f"狀態評估: is_resolved={evaluation.is_resolved}")
    if not evaluation.is_resolved:
        print(f"下一步假設: {evaluation.next_search_request.hypothesis if evaluation.next_search_request else '無'}")
        print(f"下一步行動: {evaluation.next_search_request.action_type if evaluation.next_search_request else '無'}")
        
    return {
        # 將 Pydantic 物件轉成 dict 存入 state (相容性較好)
        "current_request": evaluation.next_search_request.model_dump() if evaluation.next_search_request else None,
        "is_resolved": evaluation.is_resolved,
        "final_report": evaluation.final_report or ""
    }


def detective_node(state: GraphState):
    print(f"\n[Detective Node] 啟動 (第 {state['iterations']} 次迭代)")
    
    # 從 Engineer 的需求單中提取資訊，如果沒有就使用最初的 log clues
    if state.get("current_request"):
        req = state["current_request"]
        search_query = (
            f"【目標】: {req.get('target_value')}\n"
            f"【任務類型】: {req.get('action_type')}\n"
            f"【工程師的假設】: {req.get('hypothesis')}\n"
            f"【關注點】: {req.get('focus_point')}"
        )
    else:
        search_query = f"請根據以下 Log 線索自由檢索：{state['log_clues']}"
    
    # 給探員的專屬 Prompt (補齊了 steps 和 logs，讓它有全局 Context)
    detective_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個精準的程式碼檢索探員。
        你的任務是根據工程師的需求單，使用適當的工具去尋找程式碼。
        
        【嚴格守則】
        1. 誠實回報：如果工具回報錯誤（例如檔案找不到、路徑錯誤、無搜尋結果），請「原封不動」回傳錯誤訊息給工程師，絕對不要編造程式碼或隱瞞錯誤！
        2. 精簡輸出：找到程式碼後，只需整理出「檔案名稱」、「行號」與「完整的程式碼片段」。不需要長篇大論解釋，工程師會自己看。
        3. 嚴格文字回報：請用自然語言回報你找到的內容，絕對不要直接回傳 JSON 格式的原始資料。"""),
        ("human", "【情報需求單】\n{query}\n\n【原始 Bug 步驟】\n{steps}\n\n【原始 Log】\n{logs}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 建立 Tool Agent (請確保 global 變數 llm 和 tools 有正確定義)
    repo_dir = config["Default"]["RepoDir"]
    db_dir = config["Default"]["FAISSDBDir"]
    ensemble_retriever = build_or_load_retriever(repo_dir, db_dir, embeddings)
    tools = create_agent_tools(repo_dir=repo_dir, ensemble_retriever=ensemble_retriever)
    agent = create_tool_calling_agent(llm, tools, detective_prompt)
    agent_runner = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
    
    # 執行檢索
    result = agent_runner.invoke({
        "query": search_query,
        "steps": state["steps"],
        "logs": state["logs"]
    })
    
    # 建立這次調查的結構化報告字串，並打包進陣列
    new_report = (
        f"=== 第 {state['iterations']} 次調查 ===\n"
        f"📥 收到指令:\n{search_query}\n"
        f"📤 調查結果:\n{result['output']}\n"
    )
    
    return {
        "investigation_history": [new_report], # 使用 operator.add 會自動把這個陣列接在舊紀錄後面
        "iterations": state["iterations"] + 1
    }

# --- 路由判斷邏輯 ---
def should_continue(state: GraphState):
    # 如果已經解決，或者迭代次數超過上限 (例如 5 次)，就走向終點
    if state.get("is_resolved", False) or state["iterations"] >= 5:
        return "end"
    # 否則，退回給 Detective 繼續找資料
    return "continue"

# --- 組合與編譯 LangGraph ---
def build_debugging_graph():
    workflow = StateGraph(GraphState)
    
    # 1. 註冊 Nodes
    workflow.add_node("detective", detective_node)
    workflow.add_node("engineer", engineer_node)
    
    # 2. 定義流程順序 (Edges)
    workflow.set_entry_point("detective") # 入口點
    workflow.add_edge("detective", "engineer") # Detective 找完資料，必定交給 Engineer
    
    # 3. 條件路由 (Conditional Edge)
    workflow.add_conditional_edges(
        "engineer",
        should_continue,
        {
            "continue": "detective", # 如果還沒解決，走回 detective
            "end": END                 # 如果解決了，走向 END
        }
    )
    
    return workflow.compile()

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
    bug_reports = read_bug_report(config["Default"]["BugReportDir"])
    
    # 建立 Graph
    debugger_app = build_debugging_graph()

    for report in bug_reports:
        print(f"開始分析 bug_id: {report.bug_id}")
        print("--- 階段一：啟動 Log 解析 ---")
        clues = parse_report(report)
        print(f"萃取線索: {clues}\n")
        
        # 初始化 State
        initial_state = {
            "bug_id": report.bug_id,
            "steps": report.steps_to_reproduce,
            # 將這裡的反斜線替換掉，保護 Agentic 流程的 JSON 解析
            "logs": "\n".join([f"=== {k} ===\n{v}" for k, v in report.logs.items()]).replace("\\", "/"),
            "log_clues": clues.model_dump_json(),
            
            # 變數更名
            "investigation_history": [],
            "current_request": None, 
            
            "iterations": 1,
            "is_resolved": False,
            "final_report": ""
        }
        
        # 執行 Graph 迴圈
        # .invoke 會一直跑到抵達 END 節點才會回傳最終的 State
        final_state = debugger_app.invoke(initial_state)
        print("\n🏆 --- 最終 Bug 報告 --- 🏆")
        print(final_state.get("final_report", "無法在迭代次數內找到完整的 Root Cause。以下是目前分析：\n" + final_state.get("missing_information", "")))

        #print("--- 階段二 & 三：啟動 Agentic 迭代排查 ---")
        #final_report = run_debugging_agent(report, clues)
        #print("\n--- 最終 Bug 報告 ---")
        #print(final_report)
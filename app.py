import os
import configparser
import time
import json
import re
from typing import List, Dict, Optional
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field, model_validator
from pydantic import ValidationError
from pathlib import Path

import operator

# 現代 LangChain 工具與 LCEL
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from langchain_google_genai import ChatGoogleGenerativeAI

from tools import *
from database import build_or_load_retriever

IGNORE_DIRS = {'.git', '.vscode', 'build', 'venv', '.venv', 'dist', '.ipynb_checkpoints'}

# 創建 configparser 物件
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

# ==========================================
# 結構化資料模型 (Pydantic Models)
# ==========================================
class SingleBugReport(BaseModel):
    """單一 Bug 的資料結構"""
    bug_id: str = Field(default="", description="Bug 的唯一識別碼，例如: 'bug_000001'")
    steps_to_reproduce: str = Field(default="", description="記錄重現步驟，通常為 bug_xxxxxx.txt 內的文字內容")
    logs: Dict[str, str] = Field(default_factory=dict, description="紀錄該 bug 底下所有的 log，格式為 { 'application.log_1': 'log內容...' }")

# 定義預期的 Log 結構
class TraceFrame(BaseModel):
    file_name: str = Field(default="", description="Log 中提到的檔案路徑或名稱")
    line_number: int = Field(default=-1, description="對應的行號")

class LogClues(BaseModel):
    error_type: Optional[str] = Field(default=None, description="明確的錯誤類型...")
    file_name: Optional[str] = Field(default=None, description="錯誤發生的主要檔案路徑...")
    line_number: Optional[int] = Field(default=-1, description="錯誤發生的行號...")
    semantic_issue: str = Field(default="", description="將 Log 的行為總結為一句...")
    execution_trace: List[TraceFrame] = Field(
        default_factory=list, 
        description="從 Log 推導出的前三層執行軌跡 (Stack Trace)，由發生問題的最深層往外推。"
    )

class DetectiveCommand(BaseModel):
    """工程師開給探員的情報需求單"""
    hypothesis: str = Field(
        default="", 
        description="目前正在驗證的假設，例如：'我懷疑 user_login 函數沒有正確處理 null 值'"
    )
    action_type: str = Field(
        default="", 
        description="請填寫本次檢索的核心意圖，請從以下 4 種分類中選擇最適合的一項填入：\n"
                    "1. 'TRACE_STATE': 追蹤特定「變數」或「屬性」的讀取與修改軌跡。\n"
                    "2. 'INSPECT_LOGIC': 閱讀已知「函數」、「類別」或「具體行號」的內部實作邏輯。\n"
                    "3. 'SEARCH_CONCEPT': 透過報錯 Log、字串或口語描述，在全域尋找異常落點。\n"
                    "4. 'ANALYZE_STRUCTURE': 【警告】僅限調查 OOP 繼承關係、虛擬函數覆寫，或尋找【未知成員變數名稱】時使用。嚴禁為了解決一般邏輯錯誤而選用此項！"
    )
    target_value: str = Field(
        default="", 
        description="對應的搜尋目標，例如：變數名稱 (m_buttons)、函數名稱、檔案路徑或一段 Log 描述"
    )
    focus_point: str = Field(
        default="", 
        description="請探員特別注意什麼？例如：'請找出這個變數在哪裡被賦值'、'檢查是否有 override'"
    )

class EngineerEvaluation(BaseModel):
    """工程師的深度分析與決策報告"""
    step_by_step_reasoning: str = Field(default="", description="請詳細推演你的邏輯鏈條：上一步看到了什麼？符合預期嗎？接下來要驗證什麼？")
    is_resolved: bool = Field(default=False, description="是否已經找到 Root Cause 並能提出修復方案？")
    next_search_request: Optional[DetectiveCommand] = Field(default=None, description="【極度重要】當 is_resolved 為 false 時，此欄位【絕對必填】！請務必填寫下一步的檢索指令。只有當 is_resolved 為 true 時才允許留空 (null)。")
    final_report: Optional[str] = Field(default=None, description="如果 is_resolved 為 True，輸出完整修復報告；否則留空")
    @model_validator(mode='after')
    def check_request_if_not_resolved(self) -> 'EngineerEvaluation':
        # 如果尚未解決，但 LLM 卻沒有給下一步指令，強制報錯
        if not self.is_resolved and not self.next_search_request:
            raise ValueError("當 is_resolved 為 false 時，next_search_request 絕對不能為空！")
        return self


# ==========================================
# 全局狀態 (GraphState)
# ==========================================
class GraphState(TypedDict):
    bug_id: str
    steps: str
    logs: str
    log_clues: str
    investigation_history: List[str]
    
    # 存放 Engineer 開出的具體指令 (DetectiveCommand 的 dict 格式)
    current_request: Optional[dict] 
    
    iterations: int
    is_resolved: bool
    final_report: str
    fix_count: int

#==============================================================================================================
class TokenTrackerCallback(BaseCallbackHandler):
    def __init__(self):
        # 用於單次 Bug 分析的統計
        self.current_prompt_tokens = 0
        self.current_completion_tokens = 0
        self.current_total_tokens = 0
        
        # 用於所有 Bug 的全局累計
        self.all_prompt_tokens = 0
        self.all_completion_tokens = 0
        self.all_total_tokens = 0

    def reset_current(self):
        """在進入下一個 Bug 迴圈前，清空當前統計"""
        self.current_prompt_tokens = 0
        self.current_completion_tokens = 0
        self.current_total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        # 捕捉每次 LLM 回應的 token 消耗
        for generation in response.generations[0]:
            if hasattr(generation, 'message') and hasattr(generation.message, 'usage_metadata'):
                usage = generation.message.usage_metadata
                if usage:
                    in_tokens = usage.get('input_tokens', 0)
                    out_tokens = usage.get('output_tokens', 0)
                    total = usage.get('total_tokens', 0)
                    
                    # 累加到單次 Bug 的變數
                    self.current_prompt_tokens += in_tokens
                    self.current_completion_tokens += out_tokens
                    self.current_total_tokens += total
                    
                    # 同時累加到全局總計的變數
                    self.all_prompt_tokens += in_tokens
                    self.all_completion_tokens += out_tokens
                    self.all_total_tokens += total
#==============================================================================================================
class LoggingOutputFixingParser(OutputFixingParser):
    # 增加一個類別屬性來記錄次數
    fix_count: int = 0

    def parse(self, text: str):
        try:
            # 1. 先嘗試使用原始的 base_parser 解析
            return self.parser.parse(text)
        except (Exception, OutputParserException) as e:
            # 2. 如果捕捉到錯誤 (包含 Pydantic 的 ValueError)，代表觸發了修復機制
            self.fix_count += 1
            print(f"\n⚠️ [Parser 修復觸發] 第 {self.fix_count} 次啟動 LLM 格式修復！")
            print(f"❌ 攔截到的原始錯誤: {e}")
            print(f"🔄 正在將錯誤訊息與錯誤 JSON 送回給 LLM 重新生成...\n")
            
            # 3. 呼叫父類別原本的修復邏輯 (這會真正呼叫 LLM 進行修正)
            return super().parse(text)
#==============================================================================================================
rag_agent_system_prompt = """你是一個專為「AI 檢索探員 (RAG Agent)」提供精確搜索彈藥的【資深技術日誌萃取引擎】。
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
   
4. 執行軌跡 (execution_trace):
   - 若 Log 包含 Stack Trace 或多筆時序紀錄，請由下而上 (或依發生順序) 萃取最接近錯誤點的前 3 層檔案名稱與行號。
   - 這些軌跡將作為工程師推演「問題是如何一步步發生」的重要依據。

請保持冷靜、客觀，忽略無關痛癢的對話，輸出最精煉的技術線索。"""

# =================================================================================
engineer_system_prompt = """你是一位頂尖的資深軟體工程師，負責帶領團隊進行複雜系統的除錯 (Debugging)。
你的唯一目標是找出系統 Bug 的根本原因 (Root Cause) 並提出精確的修復方案。

你目前正在與一位「檢索探員 (Detective)」合作。探員配備了多種檢索工具，你負責提供假設並指揮他。

【🎯 核心指揮戰略 (Command Strategy)】
1. 巨觀探索 (Macro)：當只有語意線索時，請下令使用 semantic_code_search 或 exact_keyword_search 找出相關邏輯落點。
2. 函數邏輯解析 (Micro)：當你已經知道具體的檔案、函數或類別名稱，且需要閱讀「內部執行邏輯」時，請明確指示探員使用 read_function_by_line 或 read_symbol_code。
3. 變數與狀態追蹤 (State Tracking)：當你懷疑某個「變數」或「狀態」不同步時，請【絕對優先】下令使用 find_symbol_references 來追蹤該變數在哪裡被讀取或修改，切勿要求探員去盲目閱讀大量無關的函數。
1. 物件導向架構分析 (OOP Analysis)：【絕對禁止】將此作為常規的程式碼讀取手段！只有當你懷疑 Bug 與「繼承關係 (Inheritance)」、「多型 (Polymorphism)」、「找不到父類別方法」，或是極度需要一覽「所有成員變數清單」時，才能下令使用 analyze_class_architecture。
5. 推進邏輯：如果探員上一輪回報某個方向失敗，下一輪絕對不可以叫探員查一樣的地方。指令必須越來越微觀。

【💡 工作模式：假設驅動 (Hypothesis-Driven)】
1. 觀察：仔細閱讀使用者的 Bug 重現步驟與原始系統 Log。
2. 回顧：檢視歷史調查紀錄，留意上一次的假設是否被推翻。
3. 推理：將 Log 線索與探員帶回的程式碼進行交叉比對，找出邏輯斷層。
4. 行動：開立情報需求單 (DetectiveCommand) 給探員。請在 `action_type` 欄位精確填寫建議使用的工具名稱。

【🚨 核心守則】
1. 絕不憑空捏造未被檢索出來的程式碼邏輯。
2. 只針對導致「當前 Bug Log」的程式碼進行排查。
3. 終止條件：只要你已經明確知道 Bug 發生在哪個檔案、哪一行，且能寫出具體的修復程式碼 (Fix)，就【必須】將 is_resolved 設為 true 並輸出 final_report，絕對禁止再指派毫無意義的確認任務給探員。

【📝 輸出要求與 JSON 防呆守則 (極度重要)】
1. 如果你需要於 reasoning 或 hypothesis 欄位中「引用任何程式碼或 Log 內容」，請務必將原程式碼的「雙引號 (\")」替換為「單引號 (')」！
2. 絕對不可以在 JSON 的字串值內部直接出現未跳脫的雙引號。

{format_instructions}
"""

engineer_human_prompt = """
【Bug 重現步驟】
{steps}

【結構化 Log 線索 (Log Clues)】
{log_clues}

【歷史調查紀錄 (Investigation History)】
{investigation_history}

====================================
請基於以上資訊，進行深度思考並給出你的評估：
1. 如果你需要更多線索，請填寫 next_search_request 派發任務給探員。
2. 如果你已經確定根本原因，請將 is_resolved 設為 true，並給出 final_report。
"""

# =================================================================================
detective_system_prompt = """你是一個精準且高效的程式碼檢索探員。
你的任務是解讀工程師的需求單，並自主選擇最適合的工具去 Codebase 撈取情報。

【🛠️ 工具選擇 SOP (極度重要)】
工程師會建議你使用某個工具 (action_type)，但你必須根據當下情況做出最聰明的判斷：
1. 遇到「檔案與行號」：優先使用 `read_function_by_line`。只有當該工具回報找不到邊界時，你才可以降級使用 `read_code_snippet` 讀取附近小範圍的程式碼。
2. 遇到「變數追蹤」：如果工程師想查某個「變數」的定義或修改軌跡，你【必須且只能】使用 `find_symbol_references`。絕對禁止為了找變數去呼叫 `analyze_class_architecture` 或大範圍的 `read_code_snippet`。
3. 遇到「類別繼承」：追蹤物件導向的多型問題時，優先使用 `analyze_class_architecture` 釐清父子類別關係，再用 `find_virtual_overrides` 找實作。

【🚨 探員嚴格守則】
1. 禁止盲目掃描：絕不可以使用 `read_code_snippet` 一次讀取超過 50 行的程式碼！如果需要大範圍閱讀，代表你的檢索策略錯了，請改用語意搜尋或 Reference 追蹤。
2. 誠實回報：如果工具回報錯誤（例如檔案找不到、路徑錯誤），請「原封不動」回傳錯誤訊息給工程師，絕對不要編造程式碼。
3. 推進式檢索：在你的自主迴圈中，如果發現查錯方向，請立即換工具或換關鍵字，不要在同一個錯誤點死胡同裡連續呼叫相同的工具。
4. 嚴格文字回報：請用自然語言精簡回報你找到的內容（檔案、行號、程式碼重點），絕對不要直接回傳 JSON 格式的原始資料。
"""

# =================================================================================
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
def parse_report(llm, bug_report: SingleBugReport) -> LogClues:
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
        ("system", rag_agent_system_prompt),
        ("human", "【原始 Bug 報告與 Log 紀錄】\n{raw_log}")
    ])
    
    # 結合 LCEL 與 Structured Output 確保回傳 Pydantic 格式
    parser_chain = prompt | llm.with_structured_output(LogClues)
    try:
        logclues = parser_chain.invoke({"raw_log": combined_logs})
    except (Exception, ValidationError) as e:
        print(f"解析 JSON 或生成失敗: {e}")
        # 將原始 Log 進行截斷 (避免 Context Window 爆掉，取最後 1500 字元，因為錯誤通常在最下面)
        log_snippet = combined_logs[-1500:] if len(combined_logs) > 1500 else combined_logs
        
        # 將原始 Log 直接嵌入 semantic_issue 傳遞給 Engineer
        fallback_semantic = (
            "【系統警告】：LLM 結構化解析 Log 失敗。\n"
            "【探員行動建議】：請直接閱讀以下擷取的原始 Log，找出任何可疑的 '關鍵字'、'函數' 或 '檔案名稱'，"
            "並利用 exact_keyword_search 或 semantic_code_search 展開調查。\n\n"
            f"--- 原始 Log 節錄 ---\n{log_snippet}"
        )
        logclues = LogClues(
            error_type="LogParseError",
            file_name=None,
            line_number=-1,
            semantic_issue= fallback_semantic,
            execution_trace=[] 
        )

    
    # 將組合好的 logs 字串傳遞給 prompt 中的 {raw_log} 變數
    return logclues

# ==========================================
# LangGraph Workflow 建立 (使用閉包封裝 LLM 與 Tools)
# ==========================================

# 定義 Detective 專用的子狀態，負責存放工具呼叫的對話歷史
class DetectiveState(TypedDict):
    messages: Annotated[list, add_messages]

def build_detective_subgraph(detective_llm, tools):
    """建立純 LCEL 與 ToolNode 的檢索子圖"""
    
    # 定義 LLM 推論節點
    def detective_model_node(state: DetectiveState):
        messages = state["messages"]
        processed_messages = []
        
        # 統一使用合理的 Context Window 限制
        for i, msg in enumerate(messages):
            if i < 2: # 保留 System 與 Human Prompt
                processed_messages.append(msg)
                continue
                
            if isinstance(msg, ToolMessage):
                content_str = str(msg.content)
                # 給予統一且充足的長度 (如 8000)，不要任意把過去的工具結果折疊成 200 字元
                if len(content_str) > 8000:
                    short_content = content_str[:8000] + "\n\n...(⚠️ 程式碼過長已截斷，請指示更精確的行號或縮小範圍)..."
                    processed_messages.append(ToolMessage(content=short_content, tool_call_id=msg.tool_call_id, name=msg.name))
                else:
                    processed_messages.append(msg)
            else:
                # 其他訊息 (例如 AIMessage) 原樣保留
                processed_messages.append(msg)
                
        # 將「修剪過」的訊息陣列交給 LLM 推論
        response = detective_llm.bind_tools(tools).invoke(processed_messages)
        
        return {"messages": [response]}
        
    # 定義工具執行節點 (使用 LangGraph 原生的 ToolNode)
    tool_node = ToolNode(tools)
    
    # 建立子圖
    workflow = StateGraph(DetectiveState)
    workflow.add_node("detective_model", detective_model_node)
    workflow.add_node("detective_tools", tool_node)
    
    workflow.add_edge(START, "detective_model")
    
    # 內建的 tools_condition 會自動檢查 LLM 的回應是否有 tool_calls
    # 有則導向 tools，沒有則導向 END
    workflow.add_conditional_edges(
        "detective_model",
        tools_condition,
        {"tools": "detective_tools", END: END}
    )
    # 工具執行完後，強制作為上下文傳回給 LLM 繼續推論
    workflow.add_edge("detective_tools", "detective_model")
    
    return workflow.compile()

def build_debugging_graph(engineer_llm, detective_llm, tools):
    
    # --- [新增] 初始化 Detective 子圖 ---
    detective_subgraph = build_detective_subgraph(detective_llm, tools)

    # ==========================================
    # 核心節點 (Nodes)
    # ==========================================
    def engineer_node(state: GraphState):
        print(f"\n[Engineer Node] 開始深度分析 (第 {state['iterations']} 次迭代)...")
        
        # 將歷史紀錄從 List[str] 組裝成單一字串，讓 LLM 閱讀
        if not state.get("investigation_history"):
            history_text = "目前尚無調查紀錄，這是第一次推論。請根據萃取出的線索提出第一個假設並指派探員去檢索。"
        else:
            history_text = "\n\n".join(state["investigation_history"])
        
        # 建立 JsonOutputParser，並綁定你的 Pydantic 模型
        base_parser = JsonOutputParser(pydantic_object=EngineerEvaluation)

        fixing_parser = LoggingOutputFixingParser.from_llm(parser=base_parser, llm=engineer_llm)

        # 在 Prompt 結尾動態注入 Parser 產生的「嚴格格式說明 (format_instructions)」
        prompt = ChatPromptTemplate.from_messages([
            ("system", engineer_system_prompt),
            ("human", engineer_human_prompt)
        ])

        analysis_chain = prompt | engineer_llm | fixing_parser

        llm_input = {
            "steps": state["steps"],
            "log_clues": state["log_clues"],
            "investigation_history": history_text,
            "format_instructions": base_parser.get_format_instructions() 
        }

        evaluation = {}
        try:
            evaluation_dict = analysis_chain.invoke(llm_input)
            evaluation = EngineerEvaluation(**evaluation_dict)
            
        except (Exception, ValidationError) as e:
            print(f"⚠️ 解析 JSON 或生成失敗: {e}")
            print(f"evaluation_dict = {evaluation}")
            formatted_prompt = prompt.format_messages(**llm_input)
        
            print("\n====== [Debug] LLM 實際收到的完整 Prompt ======")
            for msg in formatted_prompt:
                # msg.type 會是 'system', 'human', 'ai' 等
                print(f"[{msg.type.upper()}]:\n{msg.content}\n")
            print("===============================================\n")

            # 發生極端例外時的 Fallback：強制指派探員繼續隨機調查，避免流程中斷
            fallback_request = DetectiveCommand(
                hypothesis="前一次 LLM 生成 JSON 崩潰，嘗試重新理解 Log",
                action_type="SEMANTIC_SEARCH",
                target_value="尋找導致崩潰的關鍵字",
                focus_point="請重新檢視檔案並提供精簡回報"
            )
            evaluation = EngineerEvaluation(
                step_by_step_reasoning="LLM JSON 解析失敗，啟動備援檢索方案。",
                is_resolved=False,
                next_search_request=fallback_request,
                final_report=None
            )
        
        print(f"狀態評估: is_resolved={evaluation.is_resolved}")
        if not evaluation.is_resolved:
            print(f"下一步假設: {evaluation.next_search_request.hypothesis if evaluation.next_search_request else '無'}")
            print(f"下一步行動: {evaluation.next_search_request.action_type if evaluation.next_search_request else '無'}")
            
        return {
            # 將 Pydantic 物件轉成 dict 存入 state (相容性較好)
            "current_request": evaluation.next_search_request.model_dump() if evaluation.next_search_request else None,
            "is_resolved": evaluation.is_resolved,
            "final_report": evaluation.final_report or "",
            "fix_count": state.get("fix_count", 0) + fixing_parser.fix_count
        }

    def detective_node(state: GraphState):
        print(f"\n[Detective Node] 啟動 (第 {state['iterations']} 次迭代)")

        req = state.get("current_request")
        if not req:
            # 理論上在 Engineer First 架構下永遠不會觸發，除非狀態被意外清空
            raise ValueError("探員未收到工程師的情報需求單 (current_request 為空)！")
        
        # 準備 Search Query 
        search_query = (
            f"【目標】: {req.get('target_value')}\n"
            f"【任務類型】: {req.get('action_type')}\n"
            f"【工程師的假設】: {req.get('hypothesis')}\n"
            f"【關注點】: {req.get('focus_point')}"
            )
        
        # 將 Prompt 轉換為標準的 Message 陣列
        system_msg = SystemMessage(content=detective_system_prompt)
        human_msg = HumanMessage(content=f"【情報需求單】\n{search_query}\n\n【原始 Bug 步驟】\n{state['steps']}\n\n【原始 Log】\n{state['logs']}")
        
        # 呼叫 Sub-Graph，取代 AgentExecutor
        recursion_limit = 20
        try:
            # 準備初始輸入狀態與一個變數來保存完整對話歷史
            input_state = {"messages": [system_msg, human_msg]}
            all_messages = list(input_state["messages"])
            
            # 新增字典用來暫存 tool_calls
            pending_tool_calls = {}
            
            for event in detective_subgraph.stream(
                input_state,
                config={"recursion_limit": recursion_limit}, # 👈 節點跳轉限制
                stream_mode="updates"                        # 👈 每次有節點更新時回傳狀態
            ):
                for node_name, state_update in event.items():
                    # 取出該節點產生的新訊息
                    new_messages = state_update.get("messages", [])
                    
                    for msg in new_messages:
                        all_messages.append(msg) # 將新訊息加入歷史紀錄以便後續流程使用
                        
                        # 處理 LLM 發出的訊息
                        if isinstance(msg, AIMessage):
                            if msg.content:
                                print(f"\n🤔 [Agent 思考]:\n{msg.content}")
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                # 🔥 將工具呼叫存入字典，Key 為 id，先不印出
                                for tc in msg.tool_calls:
                                    pending_tool_calls[tc['id']] = tc
                                    
                        # 處理工具回傳的結果
                        elif isinstance(msg, ToolMessage):
                            # 從暫存區抓出對應的呼叫紀錄
                            tc = pending_tool_calls.get(msg.tool_call_id)
                            if tc:
                                print(f"\n🛠️ [呼叫工具]: {tc['name']} | 參數: {tc['args']}")

                            print(f"👀 [工具結果]:\n{msg.content}")

            final_sub_state = {"messages": all_messages}
            
        except GraphRecursionError:
            print(f"⚠️ [警告] Detective 子圖觸發了 recursion_limit = {recursion_limit}，強制中斷！")
            # 建立一個模擬的強制中斷狀態，假裝這是 LLM 的最終回答
            fallback_message = AIMessage(content="【探員回報】工具呼叫陷入無限迴圈或超過次數上限，我無法得出最終結論。請提供更明確的指令或更改檢索策略。")
            # 將 fallback 訊息加進 all_messages，保留歷史紀錄
            all_messages.append(fallback_message)
            final_sub_state = {"messages": all_messages}
            
        except Exception as e:
            print(f"⚠️ [警告] Detective 子圖發生未知錯誤: {e}")
            fallback_message = AIMessage(content=f"【探員回報】執行過程中發生嚴重系統錯誤: {e}")
            final_sub_state = {"messages": [system_msg, human_msg, fallback_message]}
        
        # 手動萃取與格式化對話軌跡
        steps_info = ""
        final_output = "無結論"
        
        # 組裝歷史訊息給 Engineer 時，也要成對組裝
        history_tool_calls = {}
        
        for msg in final_sub_state["messages"]:
            if isinstance(msg, AIMessage):
                if not getattr(msg, 'tool_calls', None):
                    final_output = str(msg.content)
                elif getattr(msg, 'tool_calls', None):
                    for tc in msg.tool_calls:
                        history_tool_calls[tc['id']] = tc
            
            # 如果是工具回傳的結果
            elif isinstance(msg, ToolMessage):
                # 先貼上工具呼叫資訊
                tc = history_tool_calls.get(msg.tool_call_id)
                if tc:
                    steps_info += f"🛠️ 使用工具: {tc['name']} | 參數: {tc['args']}\n"
                
                # 再貼上回傳結果
                obs_str = str(msg.content)
                if len(obs_str) > 500:
                    obs_str = obs_str[:500] + "\n... (探員已讀取此片段，詳細內容已折疊以節省 Token)"
                steps_info += f"👀 觀察結果:\n{obs_str}\n\n"
        
        # 組裝回原本格式
        new_report = (
            f"=== 第 {state['iterations']} 次調查 ===\n"
            f"📥 收到指令:\n{search_query}\n"
            f"📤 最終結論:\n{final_output}\n"
        )
        
        # 將最新的報告加入歷史紀錄，並強制只保留最近 3 筆，防止 Context Window 爆掉
        current_history = state.get("investigation_history", [])
        current_history.append(new_report)
        if len(current_history) > 3:
            current_history = current_history[-3:]

        return {
            "investigation_history": current_history, # 覆蓋原本的 Append 行為
            "iterations": state["iterations"] + 1
        }

    # --- 路由判斷邏輯 ---
    def should_continue(state: GraphState):
        # 如果已經解決，或者迭代次數超過上限 (例如 5 次)，就走向終點
        if state.get("is_resolved", False) or state["iterations"] >= 5:
            return "end"
        # 否則，退回給 Detective 繼續找資料
        return "continue"
    
    #============================================================
    workflow = StateGraph(GraphState)
    
    # 1. 註冊 Nodes
    workflow.add_node("detective", detective_node)
    workflow.add_node("engineer", engineer_node)
    
    # 2. 定義流程順序 (Edges)
    workflow.set_entry_point("engineer") # 入口點
    
    # 3. 條件路由 (Conditional Edge)
    # Engineer 分析完後，決定是結束，還是發配任務給 Detective
    workflow.add_conditional_edges(
        "engineer",
        should_continue,
        {
            "continue": "detective", # 如果還沒解決，走回 detective
            "end": END                 # 如果解決了，走向 END
        }
    )

    # 4. Detective 執行完後，必定交回給 Engineer 進行下一步評估
    workflow.add_edge("detective", "engineer")
    
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

def enrich_trace_with_code(clues: LogClues, tools) -> str:
    """根據解析出的軌跡，直接從 AOT 計算好的 AST 字典中讀取：檔案、Function、行號以及使用的成員函數、成員變數"""
    if not clues.execution_trace:
        enriched_info = clues.model_dump()
        enriched_info.pop("execution_trace", None)
        return json.dumps(enriched_info, ensure_ascii=False, indent=2)
        
    # 取得路徑設定
    repo_dir = config["Default"]["RepoDir"]
    db_dir = config["Default"]["FAISSDBDir"]
    bounds_path = os.path.join(db_dir, "symbol_bounds.json")
    
    # 只需要讀取單一的 bounds 字典
    ast_dicts = get_ast_dictionaries()
    ast_data = ast_dicts["symbol_bounds"]
            
    trace_context = []
    # 只取前三層軌跡
    for i, frame in enumerate(clues.execution_trace[:3]):
        # 利用 resolve_best_repo_path 找真實路徑
        actual_path = resolve_best_repo_path(frame.file_name, repo_dir, IGNORE_DIRS)
        func_name = "Unknown"
        m_data_str = "無"
        m_func_str = "無"
        
        if actual_path:
            target_entities = None
            for dict_file_path, entities in ast_data.items():
                if os.path.normpath(dict_file_path) == os.path.normpath(actual_path):
                    target_entities = entities
                    break
                    
            if target_entities:
                for fname, bounds in target_entities.get("functions", {}).items():
                    if bounds["start_line"] <= frame.line_number <= bounds["end_line"]:
                        func_name = fname
                        
                        # 👉 直接讀取預先算好的陣列
                        m_data = bounds.get("used_member_data", [])
                        m_funcs = bounds.get("used_member_functions", [])
                        
                        m_data_str = ", ".join(m_data) if m_data else "無"
                        m_func_str = ", ".join(m_funcs) if m_funcs else "無"
                        break # 找到範圍最小的函數後跳出
                        
        trace_info = (f"【軌跡 {i+1}】 檔案: {frame.file_name} | Function: {func_name} | 行號: {frame.line_number}\n"
                      f"內部參照成員變數: {m_data_str}\n"
                      f"內部參照成員函數: {m_func_str}")
        trace_context.append(trace_info)
            
    # 將結果合併回 JSON
    enriched_info = clues.model_dump()
    enriched_info["enriched_code_trace"] = "\n".join(trace_context)
    enriched_info.pop("execution_trace", None)
    
    return json.dumps(enriched_info, ensure_ascii=False, indent=2)



def compress_repetitive_logs(raw_log: str, max_pattern_lines: int = 5, max_repeats: int = 2) -> str:
    """
    高效能版本的迴圈 Log 壓縮工具 (時間複雜度 O(N))
    
    :param raw_log: 原始 Log 字串
    :param max_pattern_lines: 要偵測的迴圈最大行數 (例如 A->B->C 就是 3 行)
    :param max_repeats: 允許重複的最大次數
    """
    if not raw_log:
        return raw_log

    lines = raw_log.split('\n')
    n = len(lines)
    result = []
    
    i = 0
    while i < n:
        match_found = False
        
        # 嘗試尋找重複的 pattern，從大範圍 (max_pattern_lines) 往小範圍 (1) 找
        for p_len in range(max_pattern_lines, 0, -1):
            if i + p_len <= n:
                # 擷取候選的 Pattern 區塊
                pattern = lines[i : i + p_len]
                
                # 計算這個 Pattern 連續出現了幾次
                repeats = 1
                idx = i + p_len
                # Python 的陣列切片比對 (lines[...] == pattern) 速度非常快
                while idx + p_len <= n and lines[idx : idx + p_len] == pattern:
                    repeats += 1
                    idx += p_len
                    
                # 如果重複次數超過允許上限
                if repeats > max_repeats:
                    # 保留第一組 Pattern
                    result.extend(pattern)
                    # 加上省略提示
                    result.append(f"[... 系統偵測到長度 {p_len} 行的無限迴圈，共重複 {repeats} 次，已自動省略 ...]")
                    
                    # 游標直接跳過所有重複的區域，這是高效能的關鍵！
                    i = idx 
                    match_found = True
                    break # 找到符合的迴圈就跳出 for，繼續往後掃描
                    
        # 如果這幾行沒有構成無限迴圈，就正常加入結果，游標前進 1 行
        if not match_found:
            result.append(lines[i])
            i += 1
            
    return '\n'.join(result)

# ==========================================
# 主程式：完整工作流整合
# ==========================================
if __name__ == "__main__":
    bug_reports = read_bug_report(config["Default"]["BugReportDir"])
    
    repo_dir = config["Default"]["RepoDir"]
    db_dir = config["Default"]["FAISSDBDir"]

    start_total_time = time.perf_counter() # 記錄開始時間
    
    # 建立統計實例
    token_tracker = TokenTrackerCallback()

    llm_json = ChatOllama(model=config["Default"]["ModelName"],
                        temperature=0.0,
                        seed=50, repeat_penalty=1.2,
                        num_ctx=16384,
                        num_predict=4096,
                        format="json", 
                        callbacks=[token_tracker]
                        )

    llm_text = ChatOllama(model=config["Default"]["ModelName"],
                        temperature=0.0,
                        seed=50, repeat_penalty=1.2,
                        num_ctx=16384,
                        num_predict=4096,
                        # 不加 format="json"
                        callbacks=[token_tracker]
                        )
    
    embeddings = OllamaEmbeddings(model = config["Default"]["EmbeddingModelName"])
    ensemble_retriever = build_or_load_retriever(repo_dir=repo_dir, db_dir=db_dir, ignore_dirs=IGNORE_DIRS, embeddings=embeddings)
    init_tools(repo_dir=repo_dir, db_dir=db_dir, ignore_dirs=IGNORE_DIRS, ensemble_retriever=ensemble_retriever)
    tools = [semantic_code_search,
            read_code_snippet,
            get_git_blame,
            exact_keyword_search,
            read_symbol_code,
            read_function_by_line,
            analyze_class_architecture,
            find_virtual_overrides,
            find_symbol_references]

    # 建立 Graph
    debugger_app = build_debugging_graph(llm_json, llm_text, tools)

    # ===== 初始化統計變數 =====
    total_bugs = len(bug_reports)
    resolved_bugs_count = 0
    bug_statistics = []
    # =================================

    for report in bug_reports:
        bug_start_time = time.perf_counter() # 記錄 bug 開始時間
        token_tracker.reset_current()
        print(f"開始分析 bug_id: {report.bug_id}")
        print("--- 階段一：啟動 Log 解析 ---")
        clues = parse_report(llm_json, report)
        enriched_clues_json = enrich_trace_with_code(clues, tools)
        print(f"萃取線索: {enriched_clues_json}\n")

        raw_logs_str = "\n".join(
            [f"=== {k} ===\n{v}" for k, v in report.logs.items()]
        ).replace("\\", "/")

        # 用正則表達式壓縮多行循環 Log
        compressed_logs = compress_repetitive_logs(raw_logs_str, max_pattern_lines=10, max_repeats=2)
        
        # 初始化 State
        initial_state = {
            "bug_id": report.bug_id,
            "steps": report.steps_to_reproduce,
            # 將這裡的反斜線替換掉，保護 Agentic 流程的 JSON 解析
            "logs": compressed_logs,
            "log_clues": enriched_clues_json,
            
            # 變數更名
            "investigation_history": [],
            "current_request": None, 
            
            "iterations": 1,
            "is_resolved": False,
            "final_report": "",
            "fix_count": 0
        }
        
        # 執行 Graph 迴圈
        # .invoke 會一直跑到抵達 END 節點才會回傳最終的 State
        final_state = debugger_app.invoke(initial_state)
        print("\n --- 最終 Bug 報告 --- ")
        print(final_state.get("final_report", "無法在迭代次數內找到完整的 Root Cause。以下是目前分析：\n" + final_state.get("missing_information", "")))

        # ===== 收集此 Bug 的迭代與解決狀態 =====
        current_is_resolved = final_state.get("is_resolved", False)
        current_iterations = final_state.get("iterations", 1)
        
        if current_is_resolved:
            resolved_bugs_count += 1
            
        bug_statistics.append({
            "bug_id": report.bug_id,
            "is_resolved": current_is_resolved,
            "iterations": current_iterations,
            "fix_count": final_state.get("fix_count", 0)
        })
        # ===============================================
        
        bug_end_time = time.perf_counter() # 記錄 bug 結束時間
        bug_total_time = bug_end_time - bug_start_time
        print(f"--- {report.bug_id} 花費時間 ---")
        print(f"\nbug 執行時間：{bug_total_time:.4f} 秒")
        # 迴圈尾聲：印出【單次 Bug】的消耗
        print(f"\n📊 --- {report.bug_id} Token 消耗 --- 📊")
        print(f"輸入量 : {token_tracker.current_prompt_tokens}")
        print(f"生成量 : {token_tracker.current_completion_tokens}")
        print(f"單次總用量 : {token_tracker.current_total_tokens}")
        print(f"🔧 LLM 觸發 JSON 修復次數 : {final_state.get('fix_count', 0)}")
        print("=" * 40)

        #print("--- 階段二 & 三：啟動 Agentic 迭代排查 ---")
        #final_report = run_debugging_agent(report, clues)
        #print("\n--- 最終 Bug 報告 ---")
        #print(final_report)
        
    end_total_time = time.perf_counter() # 記錄結束時間
    total_time = end_total_time - start_total_time
    print("--- 花費時間統計 ---")
    print(f"\n總執行時間：{total_time:.4f} 秒")
    # 當所有 Bug 都處理完畢跳出迴圈後，印出【全局總累計】
    print("\n🌍 --- 全局總 Token 消耗統計 --- 🌍")
    print(f"總輸入量 : {token_tracker.all_prompt_tokens}")
    print(f"總生成量 : {token_tracker.all_completion_tokens}")
    print(f"全局總累計用量: {token_tracker.all_total_tokens}")

    # ===== 印出整體的解蟲與迭代統計報告 =====
    print("\n📈 --- 整體解蟲統計報告 --- 📈")
    print(f"總 Bug 數量: {total_bugs}")
    print(f"成功解決數量: {resolved_bugs_count}")
    
    success_rate = (resolved_bugs_count / total_bugs) * 100 if total_bugs > 0 else 0
    print(f"整體解決率: {success_rate:.2f}%\n")
    
    print("--- 每條 Bug 的迭代詳細資訊 ---")
    for stat in bug_statistics:
        status_icon = "✅ 成功" if stat["is_resolved"] else "❌ 失敗"
        print(f"[{stat['bug_id']}] 狀態: {status_icon} | 總迭代次數: {stat['iterations']} 次 | 觸發 JSON 修復: {stat['fix_count']} 次")
    print("================================")
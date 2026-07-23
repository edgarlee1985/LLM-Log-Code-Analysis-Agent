import os
import configparser
import time
import json
from typing import List, Dict, Optional
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
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
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from langchain_google_genai import ChatGoogleGenerativeAI

from tools import create_agent_tools
from tools import resolve_best_repo_path
from database import build_or_load_retriever

IGNORE_DIRS = {'.git', '.vscode', 'build', 'venv', '.venv', 'dist'}

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
    file_name: str = Field(description="Log 中提到的檔案路徑或名稱")
    line_number: int = Field(description="對應的行號")

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
    hypothesis: str = Field(default="", description="目前正在驗證的假設，例如：'我懷疑 user_login 函數沒有正確處理 null 值'")
    # 🔴 允許工程師直接下令使用特定工具
    action_type: str = Field(default="", description="要執行的檢索動作，請直接填寫強烈建議探員使用的「工具名稱」，例如：read_symbol_code, analyze_class_architecture, semantic_code_search")
    target_value: str = Field(default="", description="對應的關鍵字、檔案路徑、類別名稱 (如 JetsonOrinDeployer) 或語意描述")
    # 🔴 提示工程師可以下達關於繼承的指令
    focus_point: str = Field(default="", description="請探員特別注意什麼？例如：'請幫我找出這個 class 繼承了哪個父類別'、'檢查是否有 override'")

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
    investigation_history: List[str]
    
    # 存放 Engineer 開出的具體指令 (DetectiveCommand 的 dict 格式)
    current_request: Optional[dict] 
    
    iterations: int
    is_resolved: bool
    final_report: str

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

你目前正在與一位「檢索探員 (Detective)」合作。探員負責深入 Codebase 撈取程式碼，而你負責指揮他。

【🎯 核心除錯戰略 (Core Debugging Strategy)】
1. 由廣入深與精準打擊 (Macro vs. Micro)：
   - [檔案與行號定位]：若你從 Log 中明確得知出錯的「檔案名稱與行號」，請務必優先使用 `read_function_by_line` 工具，探員會自動利用 AST 幫你切出該行所屬的完整函數邏輯。只有當該工具回報找不到邊界時，才降級使用 `read_code_snippet`。
   - [精準打擊]：一旦從 Log 或初步探索中「明確得知目標變數、函數或類別名稱」，請立即切換策略！優先使用 `read_symbol_code` 提取該符號的完整實作，或使用 `find_symbol_references` 追蹤引用，嚴禁繼續用 read_code_snippet 盲目讀取大量無關程式碼。
2. 減少上下文碎片化：
   - 盡量在一次指令中要求獲取完整的執行路徑，避免「查 A 函數 -> 發現裡面有 B 變數 -> 再下指令查 B 變數」的低效行為。
3. 避免重複與指令推進 (Push Forward)：
   - 如果探員上一輪已經回報某個方向失敗，你的下一輪指令【絕對不可以】再叫探員去查一樣的地方或退回起點！
   - 你必須推進邏輯：指派探員使用 `analyze_class_architecture` 工具找出父類別。指令必須越來越微觀。

4. 物件導向 (OOP) 與多型陷阱 (Polymorphism) 專屬守則：
   - [調查族譜]：當你發現目標與「類別 (Class)」或「物件」有關時，必須優先指派探員使用 `analyze_class_architecture` 了解該類別的繼承關係 (Base Classes)。
   - [父類別回溯]：如果子類別中找不到目標變數或函數，它極有可能定義在「父類別」中。請立刻要求探員去讀取父類別的實作。
   - [建構子陷阱]：在 C++ 中，在建構子 (Constructor) 內部呼叫虛擬函數 (Virtual Function) 時，不會觸發多型！它只會執行當前類別（或父類別）的實作，不會執行子類別的 Override。請特別留意 Log 是否從建構子發出。

【💡 工作模式：假設驅動 (Hypothesis-Driven)】
你必須嚴格遵循以下思考循環：
1. 觀察 (Observe)：仔細閱讀使用者的 Bug 重現步驟與原始系統 Log。
2. 回顧 (Review)：檢視你與探員的「歷史調查紀錄」。特別留意上一次的假設是否被推翻。
3. 推理 (Reasoning)：將 Log 線索與探員帶回的程式碼進行交叉比對，找出邏輯斷層。
4. 假設 (Hypothesis)：針對可能出錯的邏輯提出具體假設。
5. 行動 (Action)：如果你確信已找到 Root Cause，宣告結案並撰寫報告；若證據不足，開立情報需求單給探員。

【🚨 核心守則】
1. 絕不憑空捏造：絕對不要幻想或猜測未被檢索出來的程式碼邏輯。
2. 保持專注：只針對導致「當前 Bug Log」的程式碼進行排查。
3. 善用語意搜尋：如果你不知道具體的檔名或函數名稱，請指示探員使用語意搜尋。

【📝 輸出要求與 JSON 防呆守則 (極度重要)】
1. 請嚴格遵守下方提供的 JSON 結構格式進行輸出。
2. 如果你需要於 reasoning 或 hypothesis 欄位中「引用任何程式碼或 Log 內容」，請務必將原程式碼的「雙引號 (\")」替換為「單引號 (')」！
3. 絕對不可以在 JSON 的字串值內部直接出現未跳脫的雙引號，這會導致系統 JSON 解析器嚴重崩潰！
   ❌ 錯誤示範: "step_by_step_reasoning": "我看到程式碼呼叫了 qDebug() << "on_pushButton_clicked""
   ✅ 正確示範: "step_by_step_reasoning": "我看到程式碼呼叫了 qDebug() << 'on_pushButton_clicked'"

{format_instructions}
"""

engineer_human_prompt = """
【Bug 重現步驟】
{steps}

【歷史調查紀錄 (Investigation History)】
{investigation_history}

====================================
請基於以上資訊，進行深度思考並給出你的評估：
1. 如果你需要更多線索，請填寫 next_search_request 派發任務給探員。
2. 如果你已經確定根本原因，請將 is_resolved 設為 true，並給出 final_report。
"""

# =================================================================================
detective_system_prompt = """你是一個精準的程式碼檢索探員。
        你的任務是根據工程師的需求單，使用適當的工具去尋找程式碼。
        
        【嚴格守則】
        1. 誠實回報：如果工具回報錯誤（例如檔案找不到、路徑錯誤、無搜尋結果），請「原封不動」回傳錯誤訊息給工程師，絕對不要編造程式碼或隱瞞錯誤！
        2. 精簡輸出：找到程式碼後，只需整理出「檔案名稱」、「行號」與「完整的程式碼片段」。不需要長篇大論解釋，工程師會自己看。
        3. 嚴格文字回報：請用自然語言回報你找到的內容，絕對不要直接回傳 JSON 格式的原始資料。"""

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
    
    # 將組合好的 logs 字串傳遞給 prompt 中的 {raw_log} 變數
    return parser_chain.invoke({"raw_log": combined_logs})

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
            history_text = "目前尚無調查紀錄，這是第一次推論。請根據 Log 提出第一個假設並指派探員去檢索。"
        else:
            history_text = "\n\n".join(state["investigation_history"])
        
        # 建立 JsonOutputParser，並綁定你的 Pydantic 模型
        base_parser = JsonOutputParser(pydantic_object=EngineerEvaluation)

        fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=engineer_llm)

        # 在 Prompt 結尾動態注入 Parser 產生的「嚴格格式說明 (format_instructions)」
        prompt = ChatPromptTemplate.from_messages([
            ("system", engineer_system_prompt),
            ("human", engineer_human_prompt)
        ])

        analysis_chain = prompt | engineer_llm | fixing_parser

        llm_input = {
            "steps": state["steps"],
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
            "final_report": evaluation.final_report or ""
        }

    def detective_node(state: GraphState):
        print(f"\n[Detective Node] 啟動 (第 {state['iterations']} 次迭代)")
        
        # 準備 Search Query 
        if state.get("current_request"):
            req = state["current_request"]
            search_query = (
                f"【目標】: {req.get('target_value')}\n"
                f"【任務類型】: {req.get('action_type')}\n"
                f"【工程師的假設】: {req.get('hypothesis')}\n"
                f"【關注點】: {req.get('focus_point')}"
            )
        else:
            search_query = (
                f"這是第一次調查。請根據以下 Log 線索：{state['log_clues']}\n"
                f"【指令】：請只挑選「最可能發生問題的 1 個檔案或 1 個函數」進行檢索。不要一次查詢多個目標！"
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
            
            # 如果是工具回傳的結果 【修正在這裡：將給 Engineer 看的內容截斷為 500 字元】
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

def enrich_trace_with_code(clues: LogClues, tools) -> str:
    """根據解析出的軌跡，從 AST 與 Reference 字典中萃取：檔案、Function、行號以及使用的變數"""
    if not clues.execution_trace:
        return clues.model_dump_json(indent=2)
        
    # 取得路徑設定
    repo_dir = config["Default"]["RepoDir"]
    db_dir = config["Default"]["FAISSDBDir"]
    bounds_path = os.path.join(db_dir, "symbol_bounds.json")
    refs_path = os.path.join(db_dir, "symbol_references.json")
    
    # 1. 預先讀取兩個字典
    ast_data = {}
    if os.path.exists(bounds_path):
        with open(bounds_path, "r", encoding="utf-8") as f:
            ast_data = json.load(f)
            
    refs_data = {}
    if os.path.exists(refs_path):
        with open(refs_path, "r", encoding="utf-8") as f:
            refs_data = json.load(f)
            
    trace_context = []
    # 只取前三層軌跡
    for i, frame in enumerate(clues.execution_trace[:3]):
        # 利用 resolve_best_repo_path 找真實路徑
        actual_path = resolve_best_repo_path(frame.file_name, repo_dir, IGNORE_DIRS)
        func_name = "Unknown"
        func_bounds = None
        used_symbols = set() # 存放該函數內使用的變數/符號
        
        if actual_path:
            target_entities = None
            
            # (A) 從 bounds 字典找出對應檔案的實體
            for dict_file_path, entities in ast_data.items():
                if os.path.normpath(dict_file_path) == os.path.normpath(actual_path):
                    target_entities = entities
                    break
                    
            if target_entities:
                matched_bounds = None
                # 優先找函數 (範圍最小的)
                for fname, bounds in target_entities.get("functions", {}).items():
                    if bounds["start_line"] <= frame.line_number <= bounds["end_line"]:
                        if not matched_bounds or (bounds["end_line"] - bounds["start_line"] < matched_bounds["end_line"] - matched_bounds["start_line"]):
                            func_name = fname
                            matched_bounds = bounds
                            
                # 沒找到函數則找類別
                if not matched_bounds:
                    for cname, bounds in target_entities.get("classes", {}).items():
                        if bounds["start_line"] <= frame.line_number <= bounds["end_line"]:
                            if not matched_bounds or (bounds["end_line"] - bounds["start_line"] < matched_bounds["end_line"] - matched_bounds["start_line"]):
                                func_name = cname
                                matched_bounds = bounds
                                
                func_bounds = matched_bounds
                
            # (B) 如果有找到函數範圍，去 refs 字典撈出裡面用到的所有符號
            if func_bounds and refs_data:
                start_l = func_bounds["start_line"]
                end_l = func_bounds["end_line"]
                
                # 遍歷所有的 Reference 尋找落在這份檔案與這個行號區間的符號
                for symbol, file_refs in refs_data.items():
                    # 尋找這個符號是否有在我們目前的實際檔案路徑中被呼叫
                    for ref_file_path, lines in file_refs.items():
                        if os.path.normpath(ref_file_path) == os.path.normpath(actual_path):
                            # 檢查是否有行號落在該函數 (start_l ~ end_l) 範圍內
                            if any(start_l <= l <= end_l for l in lines):
                                # 過濾掉自己 (不要把函數名稱自己也算進去)
                                if symbol != func_name:
                                    used_symbols.add(symbol)
                            break
                            
        # 組裝精簡版的軌跡字串
        symbol_str = ", ".join(sorted(list(used_symbols))) if used_symbols else "無/無法辨識"
        trace_info = (f"🔻 【軌跡 {i+1}】 檔案: {frame.file_name} | Function: {func_name} | 行號: {frame.line_number}\n"
                      f"    ↳ 內部參照符號/變數: {symbol_str}")
        trace_context.append(trace_info)
            
    # 將結果合併回 JSON
    enriched_info = clues.model_dump()
    enriched_info["enriched_code_trace"] = "\n".join(trace_context)
    return json.dumps(enriched_info, ensure_ascii=False, indent=2)

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
                        seed=50, repeat_penalty=1.0,
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
    tools = create_agent_tools(repo_dir=repo_dir, db_dir=db_dir, ignore_dirs=IGNORE_DIRS, ensemble_retriever=ensemble_retriever)
    # 建立 Graph
    debugger_app = build_debugging_graph(llm_json, llm_text, tools)

    for report in bug_reports:
        bug_start_time = time.perf_counter() # 記錄 bug 開始時間
        token_tracker.reset_current()
        print(f"開始分析 bug_id: {report.bug_id}")
        print("--- 階段一：啟動 Log 解析 ---")
        clues = parse_report(llm_json, report)
        enriched_clues_json = enrich_trace_with_code(clues, tools)
        print(f"萃取線索: {enriched_clues_json}\n")
        
        # 初始化 State
        initial_state = {
            "bug_id": report.bug_id,
            "steps": report.steps_to_reproduce,
            # 將這裡的反斜線替換掉，保護 Agentic 流程的 JSON 解析
            "logs": "\n".join([f"=== {k} ===\n{v}" for k, v in report.logs.items()]).replace("\\", "/"),
            "log_clues": enriched_clues_json,
            
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
        
        bug_end_time = time.perf_counter() # 記錄 bug 結束時間
        bug_total_time = bug_end_time - bug_start_time
        print("--- {report.bug_id} 花費時間 ---")
        print(f"\nbug 執行時間：{bug_total_time:.4f} 秒")
        # 迴圈尾聲：印出【單次 Bug】的消耗
        print(f"\n📊 --- {report.bug_id} Token 消耗 --- 📊")
        print(f"輸入量 : {token_tracker.current_prompt_tokens}")
        print(f"生成量 : {token_tracker.current_completion_tokens}")
        print(f"單次總用量 : {token_tracker.current_total_tokens}")
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
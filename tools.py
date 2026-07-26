import os
import subprocess
from pathlib import Path
import json
from langchain_core.tools import tool

# 1. 內部全域變數
_tool_env = {
    "repo_dir": "",
    "db_dir": "",
    "ignore_dirs": set(),
    "ensemble_retriever": None,
    "symbol_bounds": {},
    "class_tree": {},
    "symbol_refs": {}
}

# 2. 提供一個「一次性初始化」的函數
def init_tools(repo_dir: str, db_dir: str, ignore_dirs: set[str], ensemble_retriever):
    """在程式啟動時呼叫一次，將依賴注入到全域環境中"""
    _tool_env["repo_dir"] = repo_dir
    _tool_env["db_dir"] = db_dir
    _tool_env["ignore_dirs"] = ignore_dirs
    _tool_env["ensemble_retriever"] = ensemble_retriever
    # 一次性載入三個字典檔案到記憶體
    bounds_path = os.path.join(db_dir, "symbol_bounds.json")
    if os.path.exists(bounds_path):
        with open(bounds_path, "r", encoding="utf-8") as f:
            _tool_env["symbol_bounds"] = json.load(f)

    tree_path = os.path.join(db_dir, "class_dependency_tree.json")
    if os.path.exists(tree_path):
        with open(tree_path, "r", encoding="utf-8") as f:
            _tool_env["class_tree"] = json.load(f)

    refs_path = os.path.join(db_dir, "symbol_references.json")
    if os.path.exists(refs_path):
        with open(refs_path, "r", encoding="utf-8") as f:
            _tool_env["symbol_refs"] = json.load(f)

def get_ast_dictionaries() -> dict:
    """提供唯讀的 AST 字典供外部模組 (如 app.py) 查詢，保護內部狀態不被竄改"""
    return {
        "symbol_bounds": _tool_env.get("symbol_bounds", {}),
        "class_tree": _tool_env.get("class_tree", {}),
        "symbol_refs": _tool_env.get("symbol_refs", {})
    }

def resolve_best_repo_path(log_path_hint: str, target_repo_dir: str, ignore_dirs: set[str]) -> str:
    """根據 Log 提供的路徑，利用最長共同目錄比對 (Longest Common Suffix) 找出真實的 Repo 路徑"""
    if not log_path_hint:
        return ""
    # 將 Log 路徑統一為 POSIX 格式以便切割
    log_parts = Path(log_path_hint.replace("\\", "/")).parts
    target_filename = log_parts[-1] # 取得最右邊的純檔名
    
    candidates = []
    # 掃描 Repo 找出所有「檔名完全相同」的檔案
    for root, dirs, files in os.walk(target_repo_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs] # 濾除不需要掃描的資料夾
        if target_filename in files:
            candidates.append(os.path.join(root, target_filename))
            
    if not candidates:
        return ""
        
    # 如果只有一個，就直接回傳，省下比對時間
    if len(candidates) == 1:
        return candidates[0]
                
    # 如果有多個同名檔案，進行從右到左的資料夾比對
    best_match = ""
    max_score = -1
    
    for cand_path in candidates:
        cand_parts = Path(cand_path).parts
        
        # 從右往左 (reversed) 逐層比對：檔名 -> 父資料夾 -> 祖父資料夾...
        score = 0
        for log_p, cand_p in zip(reversed(log_parts), reversed(cand_parts)):
            if log_p.lower() == cand_p.lower():
                score += 1
            else:
                break # 一旦遇到不同的資料夾名稱就停止計分
                
        if score > max_score:
            max_score = score
            best_match = cand_path
            
    return best_match

@tool
def semantic_code_search(query: str) -> str:
    """
    【功能】根據語意搜尋 Codebase。
    
    【使用時機】
    - 當不知道具體檔名，但知道邏輯異常時使用。
    
    【禁用時機 (絕對不要用)】
    - 無特別註明。
    
    【輸入規範】
    - query 為描述邏輯異常的查詢字串。
    """
    docs = _tool_env["ensemble_retriever"].invoke(query)
    result = []
    for d in docs:
        source = d.metadata.get('source', 'Unknown')
        content = d.page_content
        start = d.metadata.get('start_line', 'Unknown')
        end = d.metadata.get('end_line', 'Unknown')
        result.append(f"【檔案: {source} (行號 {start}-{end})】\n片段內容:\n{content}\n")
    return "\n\n---\n\n".join(result)

@tool
def read_code_snippet(file_path_hint: str, start_line: int, end_line: int) -> str:
    """
    【功能】讀取指定檔案與行號範圍內的程式碼片段。
    
    【使用時機】
    - 當已知檔案名稱與行號，但不確定與哪些函數、變數有關時使用。
    
    【禁用時機 (絕對不要用)】
    - ❌ 若你已經知道要找的變數或函數名稱，請改用 `read_symbol_code`，不要用此工具讀取大片無關程式碼浪費資源。
    
    【輸入規範】
    - file_path_hint 必須是絕對路徑或相對路徑，不可以只有純檔名，系統會自動進行智慧比對。
    """
    print(f"read_code_snippet, file_path_hint = {file_path_hint}, start_line = {start_line}, end_line = {end_line}")

    actual_path = resolve_best_repo_path(file_path_hint, _tool_env["repo_dir"], _tool_env["ignore_dirs"])

    if not actual_path:
        return f"讀取失敗: 找不到符合 '{file_path_hint}' 的檔案。請考慮改用語意搜尋 (semantic_code_search)。"

    try:
        with open(actual_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()[start_line-1:end_line]
        return "".join(lines)
    except Exception as e:
        return f"讀取檔案失敗: {e}"

@tool
def get_git_blame(file_path_hint: str, line_number: int) -> str:
    """
    【功能】查詢某行程式碼的 Git Blame，了解是誰在什麼時候修改了這行邏輯。
    
    【使用時機】
    - 當需要查詢特定檔案具體行號的 Git 歷史修改紀錄時使用。
    
    【禁用時機 (絕對不要用)】
    - 無特別註明。
    
    【輸入規範】
    - file_path_hint 必須是絕對路徑或相對路徑，不可以只有純檔名，系統會自動進行智慧比對。
    - 傳入的 line_number 必須確認沒有超過檔案總行數。
    """

    print(f"get_git_blame, file_path_hint = {file_path_hint}, line_number = {line_number}")

    actual_path = resolve_best_repo_path(file_path_hint, _tool_env["repo_dir"], _tool_env["ignore_dirs"])

    if not actual_path:
        return f"讀取失敗: 找不到符合 '{file_path_hint}' 的檔案。請考慮改用語意搜尋 (semantic_code_search)。"

    # 🛑 1. 防呆檢查：確保檔案與目錄真的存在
    if not os.path.exists(actual_path):
        return f"Git Blame 失敗: 找不到檔案 '{actual_path}'。"

    # 2. 取得檔案所在的目錄
    work_dir = os.path.dirname(actual_path)

    # 🛑 再次確認目錄是否為有效目錄 (避免預期外的檔案系統問題)
    if not os.path.isdir(work_dir):
        return f"Git Blame 失敗: 目錄 '{work_dir}' 無效。"

    # 3. 取得純檔名 (例如：main.cpp)
    base_name = os.path.basename(actual_path)

    # 4. 透過 cwd 參數指定在該檔案的目錄下執行指令
    try:
        return subprocess.check_output(
            ["git", "blame", "-L", f"{line_number},{line_number}", base_name],
            cwd=work_dir,
            text=True,
            encoding='utf-8',    # 👉 強制要求以 UTF-8 解碼 Git 的輸出
            errors='replace',    # 👉 遇到無法解碼的亂碼時，替換成  而非崩潰
            stderr=subprocess.STDOUT
        )

    except subprocess.CalledProcessError as e:
        # 如果 LLM 傳了超過檔案行數的數字，明確告訴它錯在哪裡
        if "has only" in e.output:
            return f"Git Blame 失敗: 檔案沒有第 {line_number} 行，請重新確認該檔案的總行數。"
        return f"Git Blame 失敗: {e.output}"
    except Exception as e:
        # 🛑 捕捉所有其他可能的錯誤，確保 Agent 不會崩潰
        return f"執行 Git Blame 時發生未知的系統錯誤: {str(e)}"

@tool
def exact_keyword_search(keyword: str, file_extension: str = "") -> str:
    """
    【功能】利用 git grep 在 Codebase 中精確搜尋特定字串。
    
    【使用時機】
    - 當你只有「非結構化文字」時使用，例如 Log 錯誤訊息 (如 'Database connection timeout')，或是寫死的字串。
    
    【禁用時機 (絕對不要用)】
    - ❌ 不要用這個工具來追蹤變數的 Call Stack，這會產生大量雜訊。
    - ❌ 如果要追蹤變數引用，請改用 `find_symbol_references`。
    
    【輸入規範】
    - keyword 必須是精確的子字串，不要傳入 Regular Expression。
    - 可以選擇性提供 file_extension 副檔名過濾 (例如: '.cpp' 或 '.py')。
    """
    print(f"exact_keyword_search, keyword = {keyword}, file_extension = {file_extension}")

    try:
        # 使用 git grep 是最快搜尋 repo 的方式 (假設 repo path 是一個 git repo)
        cmd = ["git", "grep", "-n", keyword]
        if file_extension:
            cmd.append(f"*{file_extension}")
        
        result = subprocess.check_output(
            cmd, 
            cwd=_tool_env["repo_dir"], 
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

@tool
def read_symbol_code(target_symbol: str, file_path_hint: str = "") -> str:
    """
    【功能】在全專案中搜尋特定「類別 (Class)」或「函數/方法 (Function/Method)」的完整內部實作程式碼。
    
    【使用時機】
    - 當你需要閱讀特定類別或函數的內部實作，且已經精確掌握該符號名稱時。
    - 不論是否知道該符號所在的檔案名稱皆可使用。
    
    【禁用時機 (絕對不要用)】
    - ❌ 本工具無法搜尋「變數 (Variable)」。
    - ❌ 若要追蹤變數在哪裡被呼叫或修改，請改用 `find_symbol_references`。
    
    【輸入規範】
    - target_symbol 必須是精確的類別或函數名稱 (例如 'hardwareMode' 或 'EdgeDevice')。
    - file_path_hint 為選填，若提供則只會搜尋該檔案；若留空則全域搜尋。
    """
    print(f"read_symbol_code, target_symbol = {target_symbol}, file_path_hint = {file_path_hint}")
    
    ast_data = _tool_env["symbol_bounds"]
    if not ast_data:
        return "系統錯誤： symbol_bounds 字典未成功載入。"

    # 若有提供檔案提示，先解析出真實路徑作為過濾條件
    target_actual_path = None
    if file_path_hint:
        target_actual_path = resolve_best_repo_path(file_path_hint, _tool_env["repo_dir"], _tool_env["ignore_dirs"])
        target_actual_path = os.path.normpath(target_actual_path) if target_actual_path else None

    matched_results = []

    # 1. 全域掃描 AST 字典尋找相符的符號
    for dict_file_path, entities in ast_data.items():
        # 如果有指定檔案，且路徑不匹配，則跳過
        if target_actual_path and os.path.normpath(dict_file_path) != target_actual_path:
            continue
            
        # 找類別
        for cls_name, bounds in entities.get("classes", {}).items():
            if target_symbol.lower() in cls_name.lower():
                matched_results.append({
                    "file": dict_file_path,
                    "type": f"Class",
                    "bounds": bounds
                })
                
        # 找函數
        for func_name, bounds in entities.get("functions", {}).items():
            if target_symbol.lower() in func_name.lower():
                matched_results.append({
                    "file": dict_file_path,
                    "type": f"Function",
                    "bounds": bounds
                })

    if not matched_results:
        scope_msg = f"檔案 '{file_path_hint}'" if file_path_hint else "全專案"
        return f"讀取失敗：在 {scope_msg} 中找不到名稱包含 '{target_symbol}' 的函數或類別。"

    # 2. 限制回傳數量避免 Context Window 爆掉
    MAX_MATCHES = 5
    results_str = [f"🔍 共找到 {len(matched_results)} 個相符符號 (最多顯示 {MAX_MATCHES} 個)：\n"]

    # 3. 根據收集到的邊界，實際去讀取檔案內容
    for match in matched_results[:MAX_MATCHES]:
        file_path = match["file"]
        sym_type = match["type"]
        bounds = match["bounds"]
        start_line = bounds["start_line"]
        end_line = bounds["end_line"]
        
        actual_path = resolve_best_repo_path(file_path, _tool_env["repo_dir"], _tool_env["ignore_dirs"])
        
        if not actual_path or not os.path.exists(actual_path):
            results_str.append(f"【{sym_type}】(位於 {file_path})\n-> 讀取檔案失敗：找不到實體檔案。")
            continue
            
        try:
            with open(actual_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            snippet = "".join(lines[start_idx:end_idx])
            
            # 從 bounds 字典中取出預先分析好的內部參照變數 (只有 Function 會有此欄位)
            used_data = bounds.get("used_member_data", [])
            hint_str = ""
            if used_data:
                hint_str = "\n--- 💡 智慧提示 (內部參照變數型別) ---\n" + "\n".join([f"- {d}" for d in used_data])

            # 將 hint_str 附加在原本的輸出結果後方
            results_str.append(f"【{sym_type}】(位於檔案 '{file_path}', 行號 {start_line}-{end_line})\n{snippet}{hint_str}")
            
        except Exception as e:
            results_str.append(f"【{sym_type}】(位於檔案 '{file_path}')\n-> 讀取檔案內容失敗: {e}")

    # 4. 若超過限制，給予 LLM 提示
    if len(matched_results) > MAX_MATCHES:
        results_str.append(f"\n... (還有 {len(matched_results) - MAX_MATCHES} 個同名結果被隱藏，請考慮加上 file_path_hint 來縮小範圍)")

    return "\n---\n\n".join(results_str)

@tool
def read_function_by_line(file_path_hint: str, line_number: int) -> str:
    """
    【功能】自動透過 AST 分析切出完整的函數或類別範圍，並回傳該落點的完整程式碼。
    
    【使用時機】
    - 當你從 Log 得知錯誤發生的「檔案名稱與具體行號」，想直接獲取包含該落點的「完整函數或類別程式碼」時使用。
    - 此工具比 `read_code_snippet` 更精準，可避免程式碼被截斷。
    
    【禁用時機 (絕對不要用)】
    - ❌ 若系統回報在 AST 字典中沒有解析紀錄，或沒有找到對應的邊界，無法精準切出函數時，請退回使用 `read_code_snippet` 工具。
    
    【輸入規範】
    - file_path_hint 必須是絕對路徑或相對路徑，系統會自動進行智慧比對。
    """
    print(f"read_function_by_line, file_path_hint = {file_path_hint}, line_number = {line_number}")
    
    # 1. 找出真實的檔案路徑
    actual_path = resolve_best_repo_path(file_path_hint, _tool_env["repo_dir"], _tool_env["ignore_dirs"])
    if not actual_path:
        return f"讀取失敗: 找不到符合 '{file_path_hint}' 的檔案。請考慮改用語意搜尋 (semantic_code_search)。"

    # 2. 讀取 AST 邊界字典
    ast_data = _tool_env["symbol_bounds"]
    if not ast_data:
        return "系統錯誤： symbol_bounds 字典未成功載入。"

    # 3. 找出對應檔案的 AST 實體 (注意路徑格式的比對)
    target_entities = None
    target_file_key = None
    for dict_file_path, entities in ast_data.items():
        if os.path.normpath(dict_file_path) == os.path.normpath(actual_path):
            target_entities = entities
            target_file_key = dict_file_path
            break
            
    if not target_entities:
        return f"讀取失敗：在 AST 字典中沒有 '{file_path_hint}' 的解析紀錄，無法精準切出函數。請退回使用 read_code_snippet 工具。"

    # 4. 尋找包含該行號的函數或類別
    matched_symbol_type = None
    matched_symbol_name = None
    matched_bounds = None
    
    # 優先找函數 (錯誤落點最有可能在函數內部)
    for func_name, bounds in target_entities.get("functions", {}).items():
        if bounds["start_line"] <= line_number <= bounds["end_line"]:
            # 處理巢狀函數 (Nested Function)：如果有多個函數包覆該行，取範圍最小的最精確
            if not matched_bounds or (bounds["end_line"] - bounds["start_line"] < matched_bounds["end_line"] - matched_bounds["start_line"]):
                matched_symbol_type = "Function"
                matched_symbol_name = func_name
                matched_bounds = bounds

    # 如果沒找到函數，退而求其次找類別 (錯誤可能發生在 class 的成員變數宣告區)
    if not matched_bounds:
        for cls_name, bounds in target_entities.get("classes", {}).items():
            if bounds["start_line"] <= line_number <= bounds["end_line"]:
                if not matched_bounds or (bounds["end_line"] - bounds["start_line"] < matched_bounds["end_line"] - matched_bounds["start_line"]):
                    matched_symbol_type = "Class"
                    matched_symbol_name = cls_name
                    matched_bounds = bounds

    if not matched_bounds:
            return f"讀取失敗：在檔案 '{file_path_hint}' 的第 {line_number} 行沒有找到任何對應的函數或類別邊界。請退回使用 read_code_snippet 工具讀取該行附近的程式碼。"

    # 5. 根據邊界擷取真實程式碼
    try:
        with open(actual_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        start_line = matched_bounds["start_line"]
        end_line = matched_bounds["end_line"]
        
        # 加上防呆，確保行號沒有超過檔案總行數
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        snippet = "".join(lines[start_idx:end_idx])
        
        # 從 matched_bounds 中提取 AOT 分析好的變數資訊
        used_data = matched_bounds.get("used_member_data", [])
        used_data_str = "\n".join([f"- {d}" for d in used_data]) if used_data else "- 無"
        
        return (
            f"🎯 成功於第 {line_number} 行定位到 【{matched_symbol_type}】\n"
            f"以下為完整的落點區塊 (行號 {start_line}-{end_line}):\n\n"
            f"{snippet}\n"
            f"--- 💡 智慧提示 (內部參照變數型別) ---\n"
            f"{used_data_str}"
        )
        
    except Exception as e:
        return f"讀取檔案失敗: {e}"

@tool
def analyze_class_architecture(class_name: str) -> str:
    """
    【功能】回傳該類別的完整架構（包含所在檔案、繼承關係、實作類別、成員變數清單與類別方法）。
    
    【使用時機】
    - 當你需要知道某個 Class 的父類別是誰、被哪些子類別實作，或是包含哪些「成員變數」時使用。
    - 如果你在追蹤某個類別的方法卻找不到時，請立刻對該類別使用此工具，查看它的「繼承自 (Base Classes)」，然後去父類別尋找該方法。
    
    【禁用時機 (絕對不要用)】
    - ❌ 如果你只是想追蹤「某個特定的變數或函數」為何出錯，絕對不要使用此工具！請改用 `find_symbol_references` 或 `read_symbol_code`，避免回傳過多無關資訊。
    
    【輸入規範】
    - class_name 必須輸入目標 Class 的名稱。
    """
    # 確保這裡的路徑與你存檔的路徑一致
    graph = _tool_env["class_tree"]
    if not graph:
        return "系統錯誤： class_dependency_tree 字典未成功載入。"
        
    if class_name not in graph:
        # 提供模糊搜尋建議，幫助 LLM 修正拼字錯誤
        similar_classes = [k for k in graph.keys() if class_name.lower() in k.lower()]
        if similar_classes:
            return f"找不到精確符合 '{class_name}' 的類別。您是指以下類別嗎？ {', '.join(similar_classes[:5])}"
        return f"架構圖中完全找不到類別 '{class_name}'。"
        
    data = graph[class_name]
    report = [f"📊 【類別架構報告】: {class_name}"]
    
    # 1. 所在檔案
    if data.get("source_files"):
        report.append(f"- 📍 所在檔案: {', '.join(data['source_files'])}")
    
    # 2. 繼承關係 (Inherits)
    if data.get("inherits"):
        report.append(f"- ⬆️ 繼承自 (Base Classes): {', '.join(data['inherits'])}")
    else:
        report.append("- ⬆️ 繼承自: 無 (Root Class)")
        
    # 3. 子類別實作 (Implemented By)
    if data.get("implemented_by"):
        report.append(f"- ⬇️ 被以下子類別實作 (Derived Classes): {', '.join(data['implemented_by'])}")
        
    # 4. 成員變數 (Composes) - 針對最新字典結構進行排版
    composes_list = data.get("composes", [])
    if composes_list:
        report.append("- 🧩 內部成員變數 (Composition):")
        
        # 限制輸出數量避免 Context Window 爆掉 (假設最多列出 20 個)
        limit = 20
        for item in composes_list[:limit]:
            var_type = item.get("type", "UnknownType")
            var_name = item.get("name", "unknown_var")
            # 排版成 C++ 的宣告風格，這對 LLM 來說最具備語意提示效果
            report.append(f"    * {var_type} {var_name};")
            
        if len(composes_list) > limit:
            report.append(f"    * ... (還有 {len(composes_list) - limit} 個成員變數未列出)")
    else:
        report.append("- 🧩 內部成員變數: 無或未偵測到")

    # 5. 類別方法 (Methods)
    methods_list = data.get("methods", [])
    if methods_list:
        report.append("- 🛠️ 類別方法 (Methods):")
        
        limit = 20
        for item in methods_list[:limit]:
            tags = []
            if item.get("is_virtual"): tags.append("virtual")
            if item.get("is_override"): tags.append("override")
            
            # 如果有 tag，就加上 [virtual, override] 的標籤
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            report.append(f"    * {item.get('name')}(){tag_str}")
            
        if len(methods_list) > limit:
            report.append(f"    * ... (還有 {len(methods_list) - limit} 個方法未列出)")
    else:
        report.append("- 🛠️ 類別方法: 無或未偵測到")
        
    return "\n".join(report)

@tool
def find_virtual_overrides(function_name: str, base_class: str) -> str:
    """
    【功能】尋找某個虛擬函數 (Virtual Function) 在哪些子類別中被覆寫 (Override)。
    
    【使用時機】
    - 當你發現程式碼透過父類別指標呼叫函數 (例如 base->doSomething())，想知道實際執行的子類別程式碼在哪裡時使用。
    
    【禁用時機 (絕對不要用)】
    - 無特別註明。

    【輸入規範】
    - function_name: 函數名稱 (例如 'paintEvent')。
    - base_class: 定義該虛擬函數的父類別名稱。
    """
    print(f"find_virtual_overrides, function_name = {function_name}, base_class = {base_class}")

    graph = _tool_env["class_tree"]
    if not graph:
        return "系統錯誤： class_dependency_tree 字典未成功載入。"
        
    if base_class not in graph:
        return f"找不到父類別 '{base_class}'，請確認名稱是否正確。"
        
    # 遞迴尋找所有子類別
    all_derived_classes = set()
    def find_derived(cls_name):
        derived = graph.get(cls_name, {}).get("implemented_by", [])
        for d in derived:
            if d not in all_derived_classes:
                all_derived_classes.add(d)
                find_derived(d) # 繼續往下找孫類別
                
    find_derived(base_class)
    
    if not all_derived_classes:
        return f"類別 '{base_class}' 沒有任何子類別。"
        
    # 檢查哪些子類別有 override 這個函數
    overridden_in = []
    for d_cls in all_derived_classes:
        methods = graph.get(d_cls, {}).get("methods", [])
        for m in methods:
            if m.get("name") == function_name: # 只比對名稱
                overridden_in.append(d_cls)
                break
                
    if not overridden_in:
        return f"在 '{base_class}' 的所有子類別中，沒有找到覆寫 '{function_name}' 的實作。"
        
    report = [f"🔍 虛擬函數追蹤: {base_class}::{function_name}"]
    report.append(f"以下子類別覆寫了此函數 (共 {len(overridden_in)} 個):")
    for cls in overridden_in:
        report.append(f"- {cls}")
        
    report.append("\n💡 提示：你可以使用 `read_symbol_code` 工具來閱讀上述子類別的詳細實作。")
    return "\n".join(report)

@tool
def find_symbol_references(symbol_name: str) -> str:
    """
    【功能】追蹤某個「變數、函數或類別」在哪些檔案的哪些行數被「呼叫」或「使用」。
    
    【使用時機】
    - 當你需要知道某個「變數、函數或類別」在哪些檔案的哪些行數被使用時。
    - 有助於追蹤變數的修改來源，或是函數的呼叫鏈 (Call Stack)。
    
    【禁用時機 (絕對不要用)】
    - 無特別註明。
    
    【輸入規範】
    - 請傳入精確的符號名稱 (例如: 'calculate_total' 或 'user_id')。
    """
    print(f"find_symbol_references, symbol_name = {symbol_name}")
    
    # 讀取剛剛在 Indexer 階段建立的字典
    ref_data = _tool_env["symbol_refs"]
    if not ref_data:
        return "系統錯誤： symbol_references 字典未成功載入。"

    bounds_data = _tool_env["symbol_bounds"]
    if not bounds_data:
        return "系統錯誤： symbol_bounds 字典未成功載入。"
        
    # 檢查符號是否存在
    if symbol_name not in ref_data:
        # 提供模糊搜尋建議，幫助 LLM 修正拼字錯誤
        similar_symbols = [k for k in ref_data.keys() if symbol_name.lower() in k.lower()][:5]
        if similar_symbols:
            return f"找不到精確符合 '{symbol_name}' 的 Reference。您是指以下符號嗎？ {', '.join(similar_symbols)}"
        return f"在專案中完全找不到 '{symbol_name}' 被使用的紀錄。"
        
    # 整理並排版輸出結果給 LLM
    files_dict = ref_data[symbol_name]
    report = [f"🔗 【Reference 搜尋結果】: '{symbol_name}'"]
    
    # 計算總共被引用的次數
    total_refs = sum(len(lines) for lines in files_dict.values())
    report.append(f"共找到 {total_refs} 處使用紀錄：")
    
    # 限制輸出數量避免 Context Window 爆掉
    file_count = 0
    for file_path, lines in files_dict.items():
        if file_count >= 15: # 最多列出 15 個檔案
            report.append(f"- ... (還有其他 {len(files_dict) - 15} 個檔案包含此引用)")
            break
        
        report.append(f"📄 檔案: {file_path}")
        
        # 1. 取得實際檔案路徑與 AST 邊界實體
        actual_path = resolve_best_repo_path(file_path, _tool_env["repo_dir"], _tool_env["ignore_dirs"])
        target_entities = None
        
        # 優先嘗試實際比對路徑
        if actual_path:
            for dict_file_path, entities in bounds_data.items():
                if os.path.normpath(dict_file_path) == os.path.normpath(actual_path):
                    target_entities = entities
                    break
        
        # 若 actual_path 對不到，退回使用字典內記載的原 file_path
        if not target_entities:
            for dict_file_path, entities in bounds_data.items():
                if os.path.normpath(dict_file_path) == os.path.normpath(file_path):
                    target_entities = entities
                    break

        # 2. 嘗試讀取檔案真實內容，擷取該行的程式碼
        all_lines = []
        if actual_path and os.path.exists(actual_path):
            try:
                with open(actual_path, 'r', encoding='utf-8', errors='replace') as f:
                    all_lines = f.readlines()
            except Exception:
                pass
        
        # 3. 將行號按 Function / Class 範圍進行分組 (Scope Grouping)
        scope_groups = {} 
        
        for line_num in lines:
            matched_symbol_type = None
            matched_signature = None
            matched_bounds = None
            
            if target_entities:
                # 優先找函數 (取範圍最小的)
                for func_name, bounds in target_entities.get("functions", {}).items():
                    if bounds["start_line"] <= line_num <= bounds["end_line"]:
                        if not matched_bounds or (bounds["end_line"] - bounds["start_line"] < matched_bounds["end_line"] - matched_bounds["start_line"]):
                            matched_symbol_type = "Function"
                            # 這裡使用建構索引時儲存的 signature
                            matched_signature = bounds.get("signature", func_name) 
                            matched_bounds = bounds
                            
                # 如果沒找到函數，其次找類別
                if not matched_bounds:
                    for cls_name, bounds in target_entities.get("classes", {}).items():
                        if bounds["start_line"] <= line_num <= bounds["end_line"]:
                            if not matched_bounds or (bounds["end_line"] - bounds["start_line"] < matched_bounds["end_line"] - matched_bounds["start_line"]):
                                matched_symbol_type = "Class"
                                matched_signature = cls_name
                                matched_bounds = bounds
            
            # 若都不符合，則歸類為 Global 範圍
            scope_key = f"{matched_symbol_type}, {matched_signature}" if matched_symbol_type else "Global"
            
            if scope_key not in scope_groups:
                scope_groups[scope_key] = []
            scope_groups[scope_key].append(line_num)
            
        # 4. 排版輸出
        for scope_key, scope_lines in scope_groups.items():
            if scope_key != "Global":
                report.append(f"  - {scope_key}")
                
            for line_num in scope_lines:
                if all_lines and 0 < line_num <= len(all_lines):
                    code_line = all_lines[line_num - 1].strip()
                    report.append(f"  - [行 {line_num}] {code_line}")
                else:
                    report.append(f"  - [行 {line_num}] (無法讀取程式碼)")
                    
        file_count += 1
        
    return "\n".join(report)


__all__ = [
    "init_tools",
    "get_ast_dictionaries",
    "semantic_code_search",
    "resolve_best_repo_path",
    "read_code_snippet",
    "get_git_blame",
    "exact_keyword_search",
    "read_symbol_code",
    "read_function_by_line",
    "analyze_class_architecture",
    "find_virtual_overrides",
    "find_symbol_references"]
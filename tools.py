import os
import subprocess
from pathlib import Path
from langchain_core.tools import tool

IGNORE_DIRS = {'.git', '.vscode', 'build', 'venv', '.venv', 'dist'}

# 定義給 Agent 使用的工具 (Tools)
def create_agent_tools(repo_dir: str, ensemble_retriever):
    """
    接收 config 與 retriever，產生帶有依賴注入的 Agent Tools。
    """

    @tool
    def semantic_code_search(query: str) -> str:
        """當不知道具體檔名，但知道邏輯異常時使用。根據語意搜尋 Codebase。"""
        docs = ensemble_retriever.invoke(query)
        result = []
        for d in docs:
            source = d.metadata.get('source', 'Unknown')
            content = d.page_content
            start = d.metadata.get('start_line', 'Unknown')
            end = d.metadata.get('end_line', 'Unknown')
            result.append(f"【檔案: {source} (行號 {start}-{end})】\n片段內容:\n{content}\n")
        return "\n\n---\n\n".join(result)
    
    def resolve_best_repo_path(log_path_hint: str, target_repo_dir: str) -> str:
        """根據 Log 提供的路徑，利用最長共同目錄比對 (Longest Common Suffix) 找出真實的 Repo 路徑"""
        if not log_path_hint:
            return ""
        # 將 Log 路徑統一為 POSIX 格式以便切割
        log_parts = Path(log_path_hint.replace("\\", "/")).parts
        target_filename = log_parts[-1] # 取得最右邊的純檔名
    
        candidates = []
        # 掃描 Repo 找出所有「檔名完全相同」的檔案
        for root, dirs, files in os.walk(target_repo_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS] # 濾除不需要掃描的資料夾
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
    def read_code_snippet(file_path_hint: str, start_line: int, end_line: int) -> str:
        """
        當已知檔案名稱與行號時，讀取特定範圍的程式碼。
        注意：file_path_hint 必須是絕對路徑或相對路徑，不可以只有純檔名，系統會自動進行智慧比對。
        """
        print(f"read_code_snippet, file_path_hint = {file_path_hint}, start_line = {start_line}, end_line = {end_line}")

        actual_path = resolve_best_repo_path(file_path_hint, repo_dir)
    
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
        查詢某行程式碼的 Git Blame，了解是誰在什麼時候修改了這行邏輯
        注意：file_path_hint 必須是絕對路徑或相對路徑，不可以只有純檔名，系統會自動進行智慧比對。
        """
    
        print(f"get_git_blame, file_path_hint = {file_path_hint}, line_number = {line_number}")
    
        actual_path = resolve_best_repo_path(file_path_hint, repo_dir)
    
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
        當已知明確的變數名稱、函數名稱或錯誤訊息關鍵字時使用。
        進行整個 Codebase 的精確字串比對（類似 grep）。
        可以選擇性提供副檔名過濾 (例如: '.cpp' 或 '.py')。
        """
        print(f"exact_keyword_search, keyword = {keyword}, file_extension = {file_extension}")

        try:
            # 使用 git grep 是最快搜尋 repo 的方式 (假設 repo path 是一個 git repo)
            cmd = ["git", "grep", "-n", keyword]
            if file_extension:
                cmd.append(f"*{file_extension}")
            
            result = subprocess.check_output(
                cmd, 
                cwd=config["Default"]["RepoDir"], 
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
    
    # 回傳所有的 tools 列表
    return [semantic_code_search, read_code_snippet, get_git_blame, exact_keyword_search]
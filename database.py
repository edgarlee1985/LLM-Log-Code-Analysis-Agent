import os
import pickle
from pathlib import Path
import json

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
        (class_specifier body: (field_declaration_list)) @class
        (struct_specifier body: (field_declaration_list)) @struct
        (function_definition) @function
        """
    else:
        return None, None, None
        
    return parser, ts_language, query_str

def extract_class_dependencies(class_node, source_bytes: bytes):
    """
    走訪 class/struct 節點，萃取類別名稱、繼承的父類別、成員變數，以及【成員方法 (包含 virtual/override 狀態)】。
    """
    class_name = "Unknown"
    dependencies = {
        "inherits": set(),
        "composes": set(),
        "methods": []  # 新增：用來存放方法與其屬性
    }

    # 1. 尋找類別名稱
    for child in class_node.children:
        if child.type in ['type_identifier', 'identifier']:
            class_name = source_bytes[child.start_byte:child.end_byte].decode('utf-8')
            break

    # 2. 遞迴走訪 class 內部
    def walk(node):
        # 繼承 (Inheritance) 捕捉邏輯
        if node.type in ['base_class', 'base_class_clause']:
            for c in node.children:
                if c.type not in ['access_specifier', 'virtual', ':', ',']:
                    base_name = source_bytes[c.start_byte:c.end_byte].decode('utf-8').strip()
                    if base_name and c.type in ['type_identifier', 'template_type', 'qualified_identifier', 'identifier']:
                        dependencies["inherits"].add(base_name)
                        
        # --- 新增：捕捉類別方法與成員變數 ---
        elif node.type in ['field_declaration', 'function_definition']:
            is_function = False
            method_name = None
            is_virtual = False
            is_override = False
            
            # 內部遞迴：尋找 declarator 並偵測修飾詞
            def analyze_member(n):
                nonlocal is_function, method_name, is_virtual, is_override
                
                # 偵測 virtual 與 override 關鍵字
                if n.type == 'virtual':
                    is_virtual = True
                elif n.type in ['virtual_specifier', 'override_specifier']:
                    is_override = True
                elif n.type == 'identifier':
                    # 防呆：某些舊版 parser 可能將 override 視為純 identifier
                    if source_bytes[n.start_byte:n.end_byte].decode('utf-8') == 'override':
                        is_override = True
                        
                # 確認是否為函數宣告
                if n.type == 'function_declarator':
                    is_function = True
                    for c in n.children:
                        if c.type in ['identifier', 'field_identifier', 'destructor_name']:
                            method_name = source_bytes[c.start_byte:c.end_byte].decode('utf-8')
                            
                for c in n.children:
                    analyze_member(c)
                    
            analyze_member(node)
            
            # 如果確認是函數，加入 methods 列表
            if is_function and method_name:
                dependencies["methods"].append({
                    "name": method_name,
                    "is_virtual": is_virtual,
                    "is_override": is_override
                })
            # 如果不是函數 (就是一般的成員變數)，執行原有的 Composition 邏輯
            elif node.type == 'field_declaration' and not is_function:
                field_type = None
                field_name = None
                
                for c in node.children:
                    if c.type in ['type_identifier', 'template_type', 'qualified_identifier', 'primitive_type']:
                        field_type = source_bytes[c.start_byte:c.end_byte].decode('utf-8')
                    else:
                        def find_identifier(n):
                            if n.type in ['field_identifier', 'identifier']:
                                return source_bytes[n.start_byte:n.end_byte].decode('utf-8')
                            for child in n.children:
                                res = find_identifier(child)
                                if res: return res
                            return None
                        
                        found_name = find_identifier(c)
                        if found_name:
                            field_name = found_name

                if field_type:
                    dependencies["composes"].add((field_type, field_name or "unknown"))
        
        # 繼續往下走訪其他節點
        for child in node.children:
            # 🛑 關鍵防呆：不要走入 function_definition 的 body，避免將函數內的區域變數誤認為類別成員
            if node.type == 'function_definition' and child.type == 'compound_statement':
                continue
            walk(child)

    walk(class_node)
    
    formatted_composes = [
        {"type": t, "name": n} for t, n in dependencies["composes"]
    ]
    
    # 回傳值新增 methods
    return class_name, list(dependencies["inherits"]), formatted_composes, dependencies["methods"]

def ast_chunk_document(doc: Document) -> list[Document]:
    """將單一檔案原始碼轉化為基於 AST 的多個區塊，並注入相依性 Metadata"""
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
        
        # 基礎 Metadata[cite: 1]
        metadata = {
            "source": source_path,
            "start_line": start_line,
            "end_line": end_line,
            "ast_type": capture_name
        }
        
        # 如果是 class 或 struct，萃取相依性
        if capture_name in ['class', 'struct']:
            cls_name, inherits, composes, methods = extract_class_dependencies(node, source_bytes)
            metadata["class_name"] = cls_name
            metadata["inherits"] = inherits
            metadata["composes"] = composes
            metadata["methods"] = methods
        
        numbered_snippet = "\n".join(
            [f"{start_line + i} | {line}" for i, line in enumerate(snippet.split('\n'))]
        )
        
        ast_docs.append(Document(page_content=numbered_snippet, metadata=metadata))
        
    return ast_docs

def build_global_dependency_tree(all_chunks: list[Document], save_dir: str):
    """掃描所有 Chunk 建立全域的 Class Dependency Tree 並儲存"""
    print("正在建構全域類別相依樹 (Global Class Dependency Tree)...")
    
    dependency_graph = {}
    
    for chunk in all_chunks:
        meta = chunk.metadata
        if meta.get("ast_type") in ['class', 'struct']:
            cls_name = meta.get("class_name")
            
            # 過濾掉無效名稱或未命名結構
            if not cls_name or cls_name == "Unknown":
                continue
                
            # 建立節點
            if cls_name not in dependency_graph:
                dependency_graph[cls_name] = {
                    "source_files": [],
                    "inherits": [],
                    "composes": [],
                    "methods": [],
                    "implemented_by": [] # 反向記錄：誰繼承了我
                }
            
            # 寫入來源檔案
            source = meta.get("source")
            if source not in dependency_graph[cls_name]["source_files"]:
                dependency_graph[cls_name]["source_files"].append(source)
                
            # 寫入相依關係 (使用 set 去重複後轉回 list)
            current_inherits = set(dependency_graph[cls_name]["inherits"])
            current_inherits.update(meta.get("inherits", []))
            dependency_graph[cls_name]["inherits"] = list(current_inherits)
            
            # 1. 將現有的 composes 轉為 set of tuples
            current_composes_set = {
                (d["type"], d["name"]) for d in dependency_graph[cls_name].get("composes", [])
            }
            
            # 2. 將新進來的 composes 也轉為 set of tuples 並加入
            new_composes = meta.get("composes", [])
            if new_composes:
                new_composes_set = {
                    (d.get("type", "unknown"), d.get("name", "unknown")) 
                    for d in new_composes if isinstance(d, dict)
                }
                current_composes_set.update(new_composes_set)
            
            # 3. 轉回 list of dicts 存回 dependency_graph
            dependency_graph[cls_name]["composes"] = [
                {"type": t, "name": n} for t, n in current_composes_set
            ]

            new_methods = meta.get("methods", [])
            if new_methods:
                current_methods = {
                    m["name"]: m for m in dependency_graph[cls_name].get("methods", [])
                }
                for m in new_methods:
                    name = m["name"]
                    if name not in current_methods:
                        current_methods[name] = m
                    else:
                        # 只要有任何一個宣告帶有 virtual 或 override，就標記為 True
                        current_methods[name]["is_virtual"] |= m["is_virtual"]
                        current_methods[name]["is_override"] |= m["is_override"]
                
                dependency_graph[cls_name]["methods"] = list(current_methods.values())

    # 建立反向關係 (誰繼承了這個 Base Class？)
    for child_cls, data in dependency_graph.items():
        for base_cls in data["inherits"]:
            if base_cls in dependency_graph:
                if child_cls not in dependency_graph[base_cls]["implemented_by"]:
                    dependency_graph[base_cls]["implemented_by"].append(child_cls)

    # 儲存為 JSON 供後續 Agent 讀取
    tree_path = os.path.join(save_dir, "class_dependency_tree.json")

    # 確保目標資料夾存在，如果已存在則忽略
    os.makedirs(save_dir, exist_ok=True)

    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(dependency_graph, f, indent=4, ensure_ascii=False)
        
    print(f"完成！相依樹已儲存至: {tree_path}")
    return dependency_graph

def build_symbol_bounds_index(all_chunks: list[Document], save_dir: str):
    """掃描所有 AST Chunk，建立【檔案路徑 -> 實體名稱 -> 行號邊界】的快速查詢字典"""
    print("正在建構 AST 邊界查詢字典 (Symbol Bounds Index)...")
    symbol_bounds_index = {}
    
    for chunk in all_chunks:
        meta = chunk.metadata
        ast_type = meta.get("ast_type")
        source = meta.get("source")
        
        if not source:
            continue

        if source not in symbol_bounds_index:
            symbol_bounds_index[source] = {"classes": {}, "functions": {}}
            
        # 針對類別
        if ast_type in ['class', 'struct']:
            cls_name = meta.get("class_name")
            if cls_name and cls_name != "Unknown":
                symbol_bounds_index[source]["classes"][cls_name] = {
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line")
                }
                
        # 針對函數 (如果你的 AST chunk 有抓取函數)
        elif ast_type == 'function':
            # 這裡簡單取第一行作為識別 (實際應用可依賴 tree-sitter 更精確的函數名稱抓取)
            func_signature = chunk.page_content.split('\n')[0].strip()
            # 取括號前的名稱作為簡略 Key
            func_name = func_signature.split('(')[0].split(' ')[-1].strip()
            if func_name:
                symbol_bounds_index[source]["functions"][func_name] = {
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "signature": func_signature
                }

    meta_path = os.path.join(save_dir, "symbol_bounds.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(symbol_bounds_index, f, indent=4, ensure_ascii=False)
        
    print(f"完成！AST 邊界字典已儲存至: {meta_path}")

def build_symbol_references(documents: list, save_dir: str):
    """
    掃描所有原始檔案的 AST，建立全域的 Symbol Reference 查詢字典。
    記錄每個變數、函數在哪些檔案的哪些行數被使用。
    """
    print("正在建構 Symbol Reference 字典 (Symbol References)...")
    reference_graph = {} # 格式: { symbol_name: { source_path: set(line_numbers) } }

    for doc in documents:
        # 統一將路徑斜線轉換為正斜線
        source_path = doc.metadata.get("source", "")
        ext = os.path.splitext(source_path)[1].lower()
        
        # 取得對應語言的 parser[cite: 3]
        parser, ts_language, _ = get_ast_parser_and_query(ext) 
        
        if not parser or not ts_language:
            continue

        source_bytes = doc.page_content.encode('utf-8')
        tree = parser.parse(source_bytes)

        # 遞迴走訪整棵樹，蒐集所有的 identifier 節點
        def walk_for_refs(node):
            # 針對變數名稱、函數名稱、類別名稱進行捕捉
            if node.type in ['identifier', 'field_identifier', 'type_identifier']:
                symbol_name = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
                
                # 過濾掉長度過短 (例如 i, j) 的無意義變數，縮小字典體積
                if len(symbol_name) >= 3: 
                    line_number = node.start_point[0] + 1
                    
                    if symbol_name not in reference_graph:
                        reference_graph[symbol_name] = {}
                    if source_path not in reference_graph[symbol_name]:
                        reference_graph[symbol_name][source_path] = set()
                        
                    reference_graph[symbol_name][source_path].add(line_number)
                    
            for child in node.children:
                walk_for_refs(child)

        walk_for_refs(tree.root_node)

    # 將 set 轉為 sorted list 以利 JSON 序列化儲存
    for symbol, files in reference_graph.items():
        for file in files:
            reference_graph[symbol][file] = sorted(list(files[file]))

    ref_path = os.path.join(save_dir, "symbol_references.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(reference_graph, f, indent=4, ensure_ascii=False)
        
    print(f"完成！Symbol Reference 字典已儲存至: {ref_path}")

def get_files_from_repo(repo_path: str, ignore_dirs: set[str]):
    """走訪資料夾，取得所有符合條件的檔案路徑"""
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        # 移除不需要掃描的資料夾
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ALLOWED_EXTENSIONS:
                file_path = os.path.join(root, file).replace('\\', '/')
                file_paths.append(file_path)
    return file_paths

def gen_faii_index_from_path(repo_path: str, db_dir: str, ignore_dirs: set[str], embeddings):
    print(f"開始掃描資料夾: {repo_path}")
    file_paths = get_files_from_repo(repo_path, ignore_dirs)
    
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

    build_symbol_references(documents, db_dir)

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

    # 呼叫建立樹狀圖的函數
    build_global_dependency_tree(all_chunks, db_dir)
    build_symbol_bounds_index(all_chunks, db_dir)
    
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


def build_or_load_retriever(repo_dir: str, db_dir: str, ignore_dirs: set[str], embeddings) -> EnsembleRetriever:
    """初始化並回傳混合檢索器 (EnsembleRetriever)"""
    bm25_path = os.path.join(db_dir, "bm25_retriever.pkl")
    faiss_index_path = os.path.join(db_dir, "index.faiss")
    
    if not (os.path.isdir(db_dir) and os.path.exists(bm25_path) and os.path.exists(faiss_index_path)):
        gen_faii_index_from_path(repo_dir, db_dir, ignore_dirs, embeddings)

    vector_db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

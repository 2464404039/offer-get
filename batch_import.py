"""
批量导入面经文档到 ChromaDB
从桌面上的面经文件夹读取所有 .md / .txt 文件并入库
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import RAGEngine
import chromadb
from config import CHROMA_PERSIST_DIR


# ── 要导入的文件夹 ──
DATA_DIRS = [
    r"C:\Users\admin\Desktop\AI_Agent面经资料库",
    r"C:\Users\admin\Desktop\面经",
]

# ── 支持的扩展名 ──
EXTENSIONS = (".md", ".txt")


def clear_old_data():
    """清空旧的 ChromaDB 集合（包括之前 eval 塞入的测试文档）"""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    for name in ["children", "parents"]:
        try:
            client.delete_collection(name)
            print(f"  已清空旧集合: {name}")
        except Exception:
            pass
    # 也清空 bm25 缓存
    bm25_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_index.json")
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
        print(f"  已清空 BM25 索引缓存")


def collect_files():
    """收集所有要导入的文件"""
    files = []
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            print(f"  跳过（不存在）: {data_dir}")
            continue
        for root, _, filenames in os.walk(data_dir):
            for fname in filenames:
                if fname.lower().endswith(EXTENSIONS) and fname != "README.md":
                    fpath = os.path.join(root, fname)
                    # 用相对路径做 filename（保留目录结构）
                    rel_path = os.path.relpath(fpath, data_dir)
                    files.append((fpath, rel_path))
    return files


def main():
    print("=" * 60)
    print("  批量导入面经文档")
    print("=" * 60)

    # ── 1. 清空旧数据 ──
    print("\n[1/3] 清空旧数据...")
    clear_old_data()

    # ── 2. 收集文件 ──
    print("\n[2/3] 收集文件...")
    files = collect_files()
    if not files:
        print("  ❌ 没有找到任何 .md 或 .txt 文件")
        sys.exit(1)

    print(f"  找到 {len(files)} 个文件:")
    for fpath, rel in files:
        size_kb = os.path.getsize(fpath) / 1024
        d = os.path.dirname(rel) or "."
        print(f"    [{size_kb:5.0f}KB] {d}/{os.path.basename(fpath)}")

    # ── 3. 逐文件导入 ──
    print("\n[3/3] 开始导入...")
    engine = RAGEngine()
    doc_id = 1
    success = 0
    errors = []

    for fpath, rel in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                print(f"  ⚠️  跳过空文件: {rel}")
                continue

            # 加上文件名和目录前缀作为标题（帮助检索）
            prefix = f"# {rel}\n\n来源文件夹: {os.path.dirname(rel)}\n\n"
            full_content = prefix + content

            engine.add_document(doc_id, rel, full_content)
            print(f"  ✅ [{doc_id:3d}] {rel}  ({len(full_content)} chars)")
            doc_id += 1
            success += 1

        except Exception as e:
            errors.append((rel, str(e)))
            print(f"  ❌ [{doc_id:3d}] {rel} -> 失败: {e}")
            doc_id += 1

    # ── 结果 ──
    print(f"\n{'=' * 60}")
    print(f"  导入完成: {success} 成功, {len(errors)} 失败")
    if errors:
        print(f"\n  失败文件:")
        for rel, err in errors:
            print(f"    - {rel}: {err}")
    print(f"{'=' * 60}")

    # ── 统计 ──
    child_col = engine._get_child_collection()
    parent_col = engine._get_parent_collection()
    try:
        print(f"\n  ChromaDB 状态:")
        print(f"    Child chunks:  {child_col.count()}")
        print(f"    Parent chunks: {parent_col.count()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

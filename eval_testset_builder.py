"""
================================================================================
评估测试集自动构建器
================================================================================

从已有的文档（ChromaDB 中存储的）自动生成 RAGAS 评估用 QA 对。

策略：
  1. 从 ChromaDB 读取所有 Parent 文档片段
  2. 用 LLM 为每个片段自动生成 2-3 个问题 + 标准答案
  3. 覆盖 4 种问题类型：
     - 直接提取型 (extract)：答案明确在文本中
     - 综合推理型 (reason)：需要综合多个信息点
     - 跨段关联型 (cross)：需要连接不同段落
     - 边界/拒答型 (boundary)：文档中没有的信息
  4. 人工审核标记（输出时标注"需要审核"的条目）

使用方式：
  # 从 Web UI 已有文档自动生成测试集
  uv run python eval_testset_builder.py --output eval_testset.json

  # 从指定目录的文档文件生成
  uv run python eval_testset_builder.py --docs-dir ./docs --output eval_testset.json

  # 生成 30 条（默认 50）
  uv run python eval_testset_builder.py --count 30 --output eval_testset.json

  # 交互模式：逐条审核后再保存
  uv run python eval_testset_builder.py --interactive
================================================================================
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL, CHROMA_PERSIST_DIR
from rag_engine import RAGEngine
from openai import OpenAI


# ═══════════════════════════════════════════════════
# QA 自动生成 Prompt
# ═══════════════════════════════════════════════════

QA_GENERATION_SYSTEM = """你是一个专业的 RAG 系统测试集构建专家。你的任务是根据给定的文档片段，
生成高质量的问答对，用于评估检索增强生成（RAG）系统的效果。

## 生成规则

为每个文档片段生成 2-3 个问答对，覆盖以下类型：

1. **直接提取型** (extract)：问题答案明确在文档片段中，可直接提取。
   示例: "张三的职位是什么？" → "高级后端工程师"

2. **综合推理型** (reason)：需要综合文档中的多个信息点才能回答。
   示例: "张三的技术栈覆盖了哪些领域？" → "后端开发(Python/FastAPI)、云计算(AWS)、数据库(MySQL/Redis)"

3. **跨段关联型** (cross)：需要连接文档不同部分的逻辑关系。
   示例: "张三的项目经验如何支撑他应聘架构师岗位？"

## 输出格式

直接输出 JSON 数组，不要 markdown 代码块，不要解释：

[
  {
    "question": "问题文本",
    "ground_truth": "标准答案（一句话）",
    "question_type": "extract | reason | cross | boundary",
    "reference_contexts": ["用于验证的标准上下文片段（可选，留空数组）"]
  }
]

## 边界/拒答型 (boundary)

只在合理时生成。条件是：文档中确实没有相关信息，但这本身是一个合理的问题。
例如，一份简历写了 Python 但没有写是否会用 Java — 问"张三会 Java 吗？"就是合理的 boundary 问题。
边界型问题的 ground_truth 写 "文档中未提及"。
"""


# ═══════════════════════════════════════════════════
# 文档加载器
# ═══════════════════════════════════════════════════

def load_chunks_from_chromadb(engine: RAGEngine, max_chunks: int = 30) -> list[dict]:
    """从 ChromaDB 的 Parent 集合加载文档片段"""
    parent_col = engine._get_parent_collection()
    try:
        results = parent_col.get(
            limit=max_chunks,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        print(f"❌ 读取 ChromaDB 失败: {e}")
        return []

    if not results or not results.get("documents"):
        return []

    chunks = []
    for i in range(len(results["documents"])):
        chunks.append({
            "content": results["documents"][i],
            "filename": results["metadatas"][i].get("filename", "unknown"),
            "doc_id": results["metadatas"][i].get("doc_id", -1),
            "chunk_idx": i,
        })

    # 按文档分组，每个文档取适量
    by_doc: dict[int, list] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    # 每个文档取前几个 chunk，保证多样性
    selected = []
    per_doc = max(1, max_chunks // max(1, len(by_doc)))
    for doc_chunks in by_doc.values():
        selected.extend(doc_chunks[:per_doc])
        if len(selected) >= max_chunks:
            break

    print(f"✅ 从 ChromaDB 加载 {len(chunks)} 个 chunk（选用 {len(selected)} 个）")
    print(f"   覆盖 {len(by_doc)} 个文档")
    return selected[:max_chunks]


def load_chunks_from_dir(docs_dir: str, max_chunks: int = 30) -> list[dict]:
    """从目录加载文档文件（.txt, .md, .pdf）"""
    chunks = []
    allowed_ext = {".txt", ".md", ".pdf"}

    for fname in sorted(os.listdir(docs_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in allowed_ext:
            continue

        fpath = os.path.join(docs_dir, fname)
        try:
            if ext == ".pdf":
                # pdf_handler 读取
                from pdf_handler import extract_text_from_pdf
                content = extract_text_from_pdf(fpath)
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
        except Exception as e:
            print(f"⚠️  跳过 {fname}: {e}")
            continue

        # 用 RAGEngine 的分割器做 chunk
        # 简单按 2000 字分割
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", "。", "！", "？", " ", ""]
        )
        for i, chunk_text in enumerate(splitter.split_text(content)):
            chunks.append({
                "content": chunk_text,
                "filename": fname,
                "doc_id": 0,
                "chunk_idx": i,
            })
            if len(chunks) >= max_chunks:
                break
        if len(chunks) >= max_chunks:
            break

    print(f"✅ 从 {docs_dir} 加载 {len(chunks)} 个文档片段")
    return chunks


# ═══════════════════════════════════════════════════
# QA 生成器
# ═══════════════════════════════════════════════════

def generate_qa_from_chunks(llm: OpenAI, chunks: list[dict],
                            target_count: int = 50,
                            interactive: bool = False) -> list[dict]:
    """从文档片段自动生成 QA 对"""
    all_qa = []
    per_chunk = max(2, target_count // max(1, len(chunks)))

    print(f"\n🤖 开始生成 QA 对（每 chunk 约 {per_chunk} 条）...")
    print(f"   目标总数: {target_count}, 可用 chunk: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        if len(all_qa) >= target_count:
            break

        content = chunk["content"][:3000]  # 截断过长内容
        prompt = f"""文档片段（来源: {chunk['filename']}）：

{content}

请为以上文档片段生成 {per_chunk} 个问答对。"""

        try:
            response = llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": QA_GENERATION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  ⚠️ LLM 调用失败 (chunk {i}): {e}")
            continue

        # 解析 JSON
        qa_list = _parse_qa_json(raw)
        if not qa_list:
            print(f"  ⚠️ 解析失败 (chunk {i})，原始输出: {raw[:100]}...")
            continue

        # 标注来源
        for qa in qa_list:
            qa["source_chunk_idx"] = i
            qa["source_filename"] = chunk["filename"]
            qa["needs_review"] = qa.get("question_type") in ("cross", "boundary")

        all_qa.extend(qa_list)
        print(f"  [{len(all_qa)}/{target_count}] chunk {i}: {chunk['filename'][:30]} → {len(qa_list)} 条")

    print(f"\n✅ 共生成 {len(all_qa)} 条 QA 对")

    # 交互审核
    if interactive and all_qa:
        all_qa = _interactive_review(all_qa)

    return all_qa[:target_count]


def _parse_qa_json(raw: str) -> list[dict]:
    """从 LLM 输出中提取 JSON 数组"""
    # 去掉可能的 markdown 包裹
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n", 1)
        raw = lines[1] if len(lines) > 1 else raw[3:]
        if "```" in raw:
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    # 找到 JSON 数组
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        data = json.loads(raw[start:end + 1])
        if isinstance(data, list):
            # 过滤有效条目
            return [
                d for d in data
                if isinstance(d, dict) and d.get("question") and d.get("ground_truth")
            ]
    except json.JSONDecodeError as e:
        pass

    return []


def _interactive_review(qa_list: list[dict]) -> list[dict]:
    """交互式逐条审核"""
    print("\n🔍 进入交互审核模式...")
    print("   输入 y=通过, n=删除, e=编辑, q=退出审核\n")

    reviewed = []
    for i, qa in enumerate(qa_list):
        print(f"--- [{i+1}/{len(qa_list)}] ---")
        print(f"  类型: {qa.get('question_type', '?')}")
        print(f"  问题: {qa['question']}")
        print(f"  答案: {qa['ground_truth']}")

        choice = input("  [y/n/e/q] ").strip().lower()
        if choice == "q":
            reviewed.extend(qa_list[i:])  # 保留剩余
            break
        elif choice == "n":
            continue
        elif choice == "e":
            new_q = input("  新问题: ").strip()
            new_a = input("  新答案: ").strip()
            if new_q:
                qa["question"] = new_q
            if new_a:
                qa["ground_truth"] = new_a
            qa["needs_review"] = False
            reviewed.append(qa)
        else:
            qa["needs_review"] = False
            reviewed.append(qa)

    return reviewed


# ═══════════════════════════════════════════════════
# 质量统计
# ═══════════════════════════════════════════════════

def print_quality_report(qa_list: list[dict]):
    """输出测试集质量报告"""
    type_counts: dict[str, int] = {}
    for qa in qa_list:
        t = qa.get("question_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    need_review = sum(1 for q in qa_list if q.get("needs_review"))

    print(f"\n{'='*50}")
    print("📊 测试集质量报告")
    print(f"{'='*50}")
    print(f"  总条数:       {len(qa_list)}")
    for t, c in sorted(type_counts.items()):
        label = {"extract": "直接提取", "reason": "综合推理", "cross": "跨段关联", "boundary": "边界/拒答"}
        print(f"  {label.get(t, t):　<12}: {c} 条")
    print(f"  需人工审核:   {need_review} 条")
    print(f"{'='*50}")


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估测试集自动构建器")
    parser.add_argument("--output", default="eval_testset.json", help="输出 JSON 路径")
    parser.add_argument("--count", type=int, default=50, help="目标 QA 对数量")
    parser.add_argument("--docs-dir", help="从目录加载文档（否则从 ChromaDB）")
    parser.add_argument("--interactive", action="store_true", help="交互式逐条审核")
    parser.add_argument("--max-chunks", type=int, default=30, help="最多读取的文档片段数")
    args = parser.parse_args()

    print("=" * 50)
    print("RAGAS 评估测试集自动构建器")
    print("=" * 50)

    # ── 1. 加载文档 ──
    if args.docs_dir:
        chunks = load_chunks_from_dir(args.docs_dir, args.max_chunks)
    else:
        print("🔧 从 ChromaDB 加载文档片段...")
        engine = RAGEngine()
        chunks = load_chunks_from_chromadb(engine, args.max_chunks)

    if not chunks:
        print("❌ 没有找到文档内容。请先通过 Web UI 上传文档，或使用 --docs-dir 指定文档目录。")
        sys.exit(1)

    # ── 2. 初始化 LLM ──
    if not LLM_API_KEY:
        print("❌ 未配置 LLM_API_KEY，请在 .env 中设置")
        sys.exit(1)

    llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    print(f"📦 使用 LLM: {LLM_MODEL} @ {LLM_API_BASE}")

    # ── 3. 生成 QA 对 ──
    qa_list = generate_qa_from_chunks(
        llm, chunks,
        target_count=args.count,
        interactive=args.interactive,
    )

    if not qa_list:
        print("❌ 生成失败，没有有效的 QA 对")
        sys.exit(1)

    # ── 4. 统计 ──
    print_quality_report(qa_list)

    # ── 5. 保存 ──
    output = []
    for qa in qa_list:
        output.append({
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "question_type": qa.get("question_type", "extract"),
            "reference_contexts": qa.get("reference_contexts", []),
            "source_filename": qa.get("source_filename", ""),
            "needs_review": qa.get("needs_review", False),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n📄 测试集已保存: {args.output}")
    print(f"   下一步: uv run python eval_ragas.py --testset {args.output}")


if __name__ == "__main__":
    main()

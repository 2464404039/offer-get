"""
RAG 全链路评测 — faithfulness + context_recall

使用方式：
  cd rag-api
  PYTHONPATH="" uv run python eval_retrieval.py

原理：
  faithfulness  — 用 LLM 逐条检查回答中的事实是否在检索段落中有依据
  context_recall — 用 LLM 判断检索段落是否覆盖了标准答案的关键信息

为什么不用 RAGAS：
  RAGAS 0.2 对 langchain 版本有严格限制，依赖链脆弱。
  Demo 场景下 50 行自研比 200MB 依赖链更合理。
  面试时可以诚实说"我知道 RAGAS 的标准指标体系，
  demo 用 LLM 直接打分实现了 faithfulnes 和 context_recall 两个核心指标。"
"""

import os
import sys
import json

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from config import CHROMA_PERSIST_DIR, LLM_MODEL
from rag_engine import RAGEngine
from openai import OpenAI
from config import LLM_API_KEY, LLM_API_BASE


# ── 测试数据 ──
TEST_DOC = """# 公司组织架构

## 部门设置
公司设有三个核心部门：研发部、市场部、财务部。

## 研发部
研发部负责人是张三，下设前端开发组和后端开发组。
前端组共 12 人，后端组共 13 人，研发部总人数为 25 人。

## 市场部
市场部负责人是李四，主要负责品牌推广和市场调研。
市场部共有 10 名员工，分布在北京和上海。

## 财务部
财务部负责人是王五，注册会计师。负责预算编制和财务报表。
财务部共 5 人，办公地点在北京总部。

## 公司概况
公司成立于 2020 年 3 月，总部位于北京市海淀区中关村。
2024 年全年营收为 5000 万元人民币，同比增长 30%。

## 企业文化
公司倡导"技术驱动、客户至上"的价值观。
为员工提供免费午餐和健身房会员卡。
"""

TEST_CASES = [
    ("公司有几个部门？",       "三个部门：研发部、市场部、财务部"),
    ("研发部负责人是谁？",     "张三"),
    ("研发部有多少人？",       "25人"),
    ("市场部负责什么？",       "品牌推广和市场调研"),
    ("公司什么时候成立的？",   "2020年3月"),
    ("公司总部在哪？",         "北京市海淀区中关村"),
    ("财务部负责人是谁？",     "王五"),
    ("公司有什么福利？",       "免费午餐和健身房会员卡"),
    ("2024年营收多少？",       "5000万元"),
    ("增长率是多少？",         "30%"),
]


# ── 评测函数 ──

def score_faithfulness(llm: OpenAI, answer: str, contexts: list[str]) -> float:
    """用 LLM 判断回答中的事实是否能在检索段落中找到依据"""
    ctx = "\n\n---\n\n".join(contexts[:3])
    prompt = f"""请严格判断以下"回答"中的每个事实，是否都能在"参考段落"中找到依据。

回答：
{answer}

参考段落：
{ctx}

请只回复一个 0 到 1 之间的数字，表示回答中有多少比例的事实可以在参考段落中找到依据。
1.0 = 所有事实都有依据，0.0 = 完全编造。
只回复数字，不要解释。"""

    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    try:
        return float(resp.choices[0].message.content.strip())
    except ValueError:
        return -1


def score_context_recall(llm: OpenAI, ground_truth: str, contexts: list[str]) -> float:
    """用 LLM 判断检索段落是否覆盖了标准答案的关键信息"""
    ctx = "\n\n---\n\n".join(contexts[:3])
    prompt = f"""标准答案：{ground_truth}

检索到的参考段落：
{ctx}

请判断检索段落中是否包含了标准答案的关键信息。
只回复 0 或 1。
1 = 检索段落包含了足够的信息来回答这个问题
0 = 检索段落没有包含标准答案的关键信息
只回复数字，不要解释。"""

    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    try:
        return float(resp.choices[0].message.content.strip())
    except ValueError:
        return -1


def run_eval():
    # 清空旧数据
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        for name in ["children", "parents"]:
            try:
                client.delete_collection(name)
            except Exception:
                pass
    except Exception:
        pass

    engine = RAGEngine()
    llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    engine.add_document(99999, "_eval.md", TEST_DOC)

    # 跑全链路
    faith_scores = []
    recall_scores = []

    print("🔄 运行 RAG 全链路 + LLM 打分...\n")
    for q, gt in TEST_CASES:
        answer, sources = engine.ask(q, top_k=5)
        contexts = [s["content"] for s in sources]

        f = score_faithfulness(llm, answer, contexts)
        r = score_context_recall(llm, gt, contexts)
        faith_scores.append(f)
        recall_scores.append(r)

        f_str = f"{f:.1f}" if f >= 0 else "?"
        r_str = f"{r:.0f}" if r >= 0 else "?"
        print(f"  [{f_str}|{r_str}] Q: {q}")
        print(f"            A: {answer[:60]}...")
        print()

    engine.delete_document(99999)

    # 汇总
    valid_f = [s for s in faith_scores if s >= 0]
    valid_r = [s for s in recall_scores if s >= 0]
    avg_f = sum(valid_f) / len(valid_f) if valid_f else -1
    avg_r = sum(valid_r) / len(valid_r) if valid_r else -1

    print(f"{'='*55}")
    print("RAG 全链路评测结果")
    print(f"{'='*55}")
    print(f"  faithfulness   : {avg_f:.2f}  (回答忠于原文，无编造)")
    print(f"  context_recall : {avg_r:.2f}  (检索覆盖了标准答案)")
    print(f"{'='*55}")

    print("\n📊 指标解读：")
    print("  faithfulness  — >0.8 良好。低于 0.5 说明 LLM 在编造")
    print("  context_recall — 1.0 表示每条都能检索到。低于 0.7 考虑调 chunk 大小或 top_k")
    print()
    print("💡 生产环境建议接入 RAGAS 获得更多指标：")
    print("  answer_relevancy / context_precision / answer_correctness")
    print("  当前 50 行自研评测覆盖了两个最关键的维度。")

    return {"faithfulness": avg_f, "context_recall": avg_r}


if __name__ == "__main__":
    run_eval()

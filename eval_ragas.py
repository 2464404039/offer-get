"""
================================================================================
RAGAS 6 指标评估 + 消融实验 + 快速回归 + 网格搜索 + 断点恢复
================================================================================

评估方法论（对应 RAGAS paper）：
  对每个指标，不做"整体印象分"，而是：
    1. Decompose — 把目标（答案/标准答案/检索结果）拆成原子单元
    2. Judge — 逐条判断每个原子单元
    3. Ratio — 算 正例数/总例数，自然产生 0.50、0.67、0.83 等连续值

  指标：
    - context_precision     逐 chunk 判相关 → relevant/total
    - context_recall        拆 ground_truth 为信息原子 → 逐条判检索覆盖 → found/total
    - faithfulness          拆回答为事实声明 → 逐条判 context 是否有依据 → verified/total
    - answer_relevancy      拆回答为句子 → 逐句判是否对答题有贡献 → helpful/total
    - answer_correctness    提取 ground_truth 关键点 → 逐条判回答是否命中 → hit/total
    - context_entity_recall 提取 ground_truth 实体 → 逐条判 context 是否包含 → found/total

  消融实验对比 4 种检索配置：
    A. 纯向量检索 (Child 级)
    B. 纯 BM25 关键词检索
    C. 混合检索 (向量 + BM25 + RRF)  你的方案
    D. 单一粒度切块 (无 Parent-Child 分层)  对比基线

  新增模式：
    --quick       快速回归（5 条种子、3 个核心指标、仅混合检索）
    --grid        网格搜索（参数组合遍历）
    --resume      断点恢复（隐式，检测已有结果自动跳过）

  使用方式：
    uv run python eval_ragas.py --testset eval_testset_sample.json --skip-ablation
    uv run python eval_ragas.py --testset eval_testset.json --limit 8 --skip-ablation
    uv run python eval_ragas.py --quick
    uv run python eval_ragas.py --quick --baseline eval_quick_result.json
    uv run python eval_ragas.py --grid "BM25_WEIGHT:0.2,0.3,0.4" "VECTOR_WEIGHT:0.5,0.6,0.7"
    uv run python eval_ragas.py --grid-file grid_params.json
    uv run python eval_ragas.py --testset eval_testset.json  (auto-resume if eval_results.json exists)
================================================================================
"""

import argparse
import importlib
import itertools
import json
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from config import (
    LLM_API_KEY, LLM_API_BASE, LLM_MODEL,
    TOP_K, BM25_WEIGHT, VECTOR_WEIGHT, K_RRF,
)
import config as config_module
from rag_engine import RAGEngine, _build_prompt
import rag_engine as rag_engine_module


# ═══════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════

@dataclass
class EvalCase:
    question: str
    ground_truth: str
    reference_contexts: list[str] = field(default_factory=list)


@dataclass
class AblationResult:
    name: str
    metrics: dict[str, float]
    num_samples: int
    elapsed_sec: float


# ═══════════════════════════════════════════════════
# JSON 解析工具
# ═══════════════════════════════════════════════════

def _extract_json(raw: str) -> dict | list | None:
    """从 LLM 回复中提取 JSON（容忍 markdown 代码块包裹）"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            text = "\n".join(lines[1:])
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    text = text.strip()
    if text.startswith("{"):
        end = text.rfind("}")
        if end != -1:
            text = text[:end + 1]
    elif text.startswith("["):
        end = text.rfind("]")
        if end != -1:
            text = text[:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _call_llm(llm: OpenAI, system: str, user: str, temperature: float = 0,
              max_tokens: int = 2048) -> str:
    """调 LLM，返回文本"""
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# ═══════════════════════════════════════════════════
# LLM Judge（Decompose → Judge → Ratio）
# ═══════════════════════════════════════════════════

class LLMJudge:
    """
    实现 RAGAS 6 指标，方法论：
      Decompose → 逐条 Judge → 算比例

    不直接让 LLM "给个分数"，而是让它拆碎了逐条判断。
    这是 RAGAS 库内部实际做的事。
    """

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    # ═══════════════════════════════════════════════════
    # 1. Context Precision — 逐 chunk 判相关
    # ═══════════════════════════════════════════════════

    def context_precision(self, question: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0

        chunks_text = ""
        for i, c in enumerate(contexts[:10]):
            chunks_text += f"\n[Chunk {i+1}]\n{c[:400]}\n"

        prompt = f"""Task: Judge whether each retrieved chunk is relevant to the question.

Question: {question}

Chunks:
{chunks_text}

For each chunk, output a JSON object mapping chunk index to "YES" or "NO".
Example: {{"1": "YES", "2": "NO", "3": "YES"}}
Output ONLY the JSON object, no explanation."""

        raw = _call_llm(self.llm, "You are a retrieval quality evaluator. Output JSON only.", prompt, max_tokens=512)
        verdicts = _extract_json(raw)

        if not isinstance(verdicts, dict):
            return -1.0

        yes = sum(1 for v in verdicts.values() if isinstance(v, str) and v.upper() == "YES")
        total = len(verdicts)
        return yes / total if total > 0 else 0.0

    # ═══════════════════════════════════════════════════
    # 2. Context Recall — 拆 ground_truth → 逐信息原子判覆盖
    # ═══════════════════════════════════════════════════

    def context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0

        ctx_text = "\n\n---\n\n".join(c[:500] for c in contexts[:5])

        decomp_prompt = f"""Task: Break the following ground truth answer into atomic, independent information units.

Ground Truth: {ground_truth}

Rules:
- Each unit must be a single, verifiable fact
- Units should be self-contained (include the subject)
- Output a JSON array of strings
Example for "3 departments: R&D, Marketing, Finance":
  ["There are 3 departments", "The departments are: R&D", "The departments are: Marketing", "The departments are: Finance"]

Output ONLY the JSON array."""

        raw = _call_llm(self.llm, "You decompose text into atomic facts. Output JSON only.", decomp_prompt, max_tokens=1024)
        atoms = _extract_json(raw)

        if not isinstance(atoms, list) or len(atoms) == 0:
            return -1.0

        atoms_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(atoms))
        verify_prompt = f"""Task: For each information unit, judge whether it can be found in or inferred from the retrieved contexts.

Retrieved Contexts:
{ctx_text}

Information Units:
{atoms_text}

For each unit, output "YES" if the information appears in the contexts, "NO" otherwise.
Output a JSON object: {{"1": "YES", "2": "NO", ...}}
Output ONLY the JSON object."""

        raw = _call_llm(self.llm, "You verify facts against source text. Output JSON only.", verify_prompt, max_tokens=512)
        verdicts = _extract_json(raw)

        if not isinstance(verdicts, dict):
            return -1.0

        yes = sum(1 for v in verdicts.values() if isinstance(v, str) and v.upper() == "YES")
        return yes / len(atoms)

    # ═══════════════════════════════════════════════════
    # 3. Faithfulness — 拆回答为 claims → 逐条判 context 有据
    # ═══════════════════════════════════════════════════

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        if not contexts or not answer.strip():
            return 0.0

        ctx_text = "\n\n---\n\n".join(c[:600] for c in contexts[:5])

        decomp_prompt = f"""Task: Break the following answer into individual factual claims.

Answer: {answer[:2000]}

Rules:
- Each claim must be a single, self-contained factual statement
- Extract ONLY factual claims (skip greetings, transitions, suggestions, hedging)
- If the answer says "The document mentions X" or "According to the document, X", extract the actual fact X
- Output a JSON array of strings
Example: "The company has 3 departments. R&D has 25 people. The culture is great."
  → ["The company has 3 departments", "R&D has 25 people", "The culture is great"]

Output ONLY the JSON array."""

        raw = _call_llm(self.llm, "You decompose answers into atomic factual claims. Output JSON only.",
                       decomp_prompt, max_tokens=1024)
        claims = _extract_json(raw)

        if not isinstance(claims, list) or len(claims) == 0:
            return -1.0

        claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        verify_prompt = f"""Task: For each claim, judge whether it is DIRECTLY supported by the source documents.

Source Documents:
{ctx_text}

Claims:
{claims_text}

For each claim, answer "YES" if the source documents contain evidence for it, "NO" otherwise.
A claim is supported ONLY if the information is explicitly present in the documents.
Vague or implied support counts as "NO".

Output a JSON object: {{"1": "YES", "2": "NO", ...}}
Output ONLY the JSON object."""

        raw = _call_llm(self.llm, "You verify claims against source text. Be strict. Output JSON only.",
                       verify_prompt, max_tokens=512)
        verdicts = _extract_json(raw)

        if not isinstance(verdicts, dict):
            return -1.0

        yes = sum(1 for v in verdicts.values() if isinstance(v, str) and v.upper() == "YES")
        return yes / len(claims)

    # ═══════════════════════════════════════════════════
    # 4. Answer Relevancy — 拆回答为句子 → 逐句判贡献
    # ═══════════════════════════════════════════════════

    def answer_relevancy(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0

        decomp_prompt = f"""Task: Split the following answer into individual sentences or independent statements.

Answer: {answer[:2000]}

Output a JSON array of strings, one per sentence/statement.
Output ONLY the JSON array."""

        raw = _call_llm(self.llm, "You split text into sentences. Output JSON only.", decomp_prompt, max_tokens=1024)
        sentences = _extract_json(raw)

        if not isinstance(sentences, list) or len(sentences) == 0:
            return -1.0

        sents_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
        verify_prompt = f"""Task: For each sentence, judge whether it contributes to answering the question.

Question: {question}

Sentences:
{sents_text}

Judge each sentence: "YES" if it provides relevant information that helps answer the question.
"NO" if it's off-topic, filler, repetition, or irrelevant.

Output a JSON object: {{"1": "YES", "2": "NO", ...}}
Output ONLY the JSON object."""

        raw = _call_llm(self.llm, "You judge relevance of sentences to questions. Output JSON only.",
                       verify_prompt, max_tokens=512)
        verdicts = _extract_json(raw)

        if not isinstance(verdicts, dict):
            return -1.0

        yes = sum(1 for v in verdicts.values() if isinstance(v, str) and v.upper() == "YES")
        return yes / len(sentences)

    # ═══════════════════════════════════════════════════
    # 5. Answer Correctness — 拆 ground_truth → 逐关键点判命中
    # ═══════════════════════════════════════════════════

    def answer_correctness(self, ground_truth: str, answer: str) -> float:
        if not answer.strip():
            return 0.0

        decomp_prompt = f"""Task: Break the following ground truth into key factual points that a correct answer MUST include.

Ground Truth: {ground_truth}

Rules:
- Extract the core, non-negotiable facts
- Each point should be independently verifiable
- Output a JSON array of strings
Example: "R&D lead is Zhang San, 25 people total" → ["R&D lead is Zhang San", "R&D has 25 people"]

Output ONLY the JSON array."""

        raw = _call_llm(self.llm, "You extract key facts from text. Output JSON only.", decomp_prompt, max_tokens=1024)
        key_points = _extract_json(raw)

        if not isinstance(key_points, list) or len(key_points) == 0:
            return -1.0

        points_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(key_points))
        verify_prompt = f"""Task: For each key point, judge whether the ANSWER correctly addresses it.

Key Points (what the answer SHOULD contain):
{points_text}

Answer to evaluate:
{answer[:2000]}

For each key point, answer:
- "YES" if the answer correctly states this fact (wording can differ)
- "PARTIAL" if the answer touches on it but is incomplete or imprecise
- "NO" if the answer does not address this point or gets it wrong

Output a JSON object: {{"1": "YES", "2": "PARTIAL", "3": "NO", ...}}
Output ONLY the JSON object."""

        raw = _call_llm(self.llm, "You compare facts in answers against ground truth. Output JSON only.",
                       verify_prompt, max_tokens=512)
        verdicts = _extract_json(raw)

        if not isinstance(verdicts, dict):
            return -1.0

        score = 0.0
        for v in verdicts.values():
            if isinstance(v, str):
                if v.upper() == "YES":
                    score += 1.0
                elif v.upper() == "PARTIAL":
                    score += 0.5
        return score / len(key_points)

    # ═══════════════════════════════════════════════════
    # 6. Context Entity Recall — 提取实体 → 逐条判出现
    # ═══════════════════════════════════════════════════

    def context_entity_recall(self, ground_truth: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0

        ctx_text = "\n\n---\n\n".join(c[:500] for c in contexts[:5])

        extract_prompt = f"""Task: Extract all key named entities from the ground truth.

Ground Truth: {ground_truth}

Extract: people names, numbers, dates, organization names, locations, technical terms, product names.
Output a JSON array of strings. Each entity should be a standalone term.
Example: "Zhang San leads R&D in Beijing with 25 people" → ["Zhang San", "R&D", "Beijing", "25"]

Output ONLY the JSON array."""

        raw = _call_llm(self.llm, "You extract named entities from text. Output JSON only.", extract_prompt, max_tokens=512)
        entities = _extract_json(raw)

        if not isinstance(entities, list) or len(entities) == 0:
            return -1.0

        entities = list(set(str(e) for e in entities))

        ents_text = "\n".join(f"{i+1}. {e}" for i, e in enumerate(entities))
        verify_prompt = f"""Task: For each entity, check whether it appears in the retrieved contexts.

Entities:
{ents_text}

Retrieved Contexts:
{ctx_text}

For each entity, answer "YES" if the entity (or a clear synonym) appears in the contexts, "NO" otherwise.
Be liberal: "Zhang San" matches "Zhang San", "张三", "Mr. Zhang", etc.

Output a JSON object: {{"1": "YES", "2": "NO", ...}}
Output ONLY the JSON object."""

        raw = _call_llm(self.llm, "You check entity presence in text. Output JSON only.", verify_prompt, max_tokens=512)
        verdicts = _extract_json(raw)

        if not isinstance(verdicts, dict):
            return -1.0

        yes = sum(1 for v in verdicts.values() if isinstance(v, str) and v.upper() == "YES")
        return yes / len(entities)

    # ═══════════════════════════════════════════════════
    # 批量评分
    # ═══════════════════════════════════════════════════

    def score_all(self, question: str, answer: str, contexts: list[str],
                  ground_truth: str) -> dict[str, float]:
        """对一条样本计算全部 6 个指标（12 次 LLM 调用）"""
        return {
            "context_precision": self.context_precision(question, contexts),
            "context_recall": self.context_recall(ground_truth, contexts),
            "faithfulness": self.faithfulness(answer, contexts),
            "answer_relevancy": self.answer_relevancy(question, answer),
            "answer_correctness": self.answer_correctness(ground_truth, answer),
            "context_entity_recall": self.context_entity_recall(ground_truth, contexts),
        }

    def score_quick(self, question: str, answer: str, contexts: list[str],
                    ground_truth: str) -> dict[str, float]:
        """快速模式：只计算 3 个核心指标（6 次 LLM 调用）"""
        return {
            "context_precision": self.context_precision(question, contexts),
            "context_recall": self.context_recall(ground_truth, contexts),
            "faithfulness": self.faithfulness(answer, contexts),
        }


def aggregate_scores(all_scores: list[dict[str, float]]) -> dict[str, float]:
    """将逐条评分聚合为平均值（跳过 -1 的无效值）"""
    aggregated = {}
    if not all_scores:
        return aggregated
    keys = all_scores[0].keys()
    for key in keys:
        valid = [s[key] for s in all_scores if s.get(key, -1) >= 0]
        aggregated[key] = sum(valid) / len(valid) if valid else -1.0
    return aggregated


# ═══════════════════════════════════════════════════
# 检索变体 — 消融实验用
# ═══════════════════════════════════════════════════

class BaselineRetrievers:
    """4 种检索器：向量 / BM25 / 混合RRF / 单一粒度"""

    def __init__(self, engine: RAGEngine):
        self.engine = engine

    def vector_only(self, query: str, top_k: int = TOP_K) -> list[dict]:
        return self.engine.search(query, top_k)

    def bm25_only(self, query: str, top_k: int = TOP_K) -> list[dict]:
        self.engine._ensure_bm25_ready()
        query_tokens = self.engine._tokenize(query)
        if self.engine._bm25_model is None:
            return []
        scores = self.engine._bm25_model.get_scores(query_tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = [i for i, _ in indexed[:top_k] if scores[i] > 0]
        parent_ids = [self.engine._bm25_doc_ids[i] for i in top]
        if not parent_ids:
            return []
        parent_col = self.engine._get_parent_collection()
        results = parent_col.get(ids=parent_ids)
        sources = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"]):
                sources.append({
                    "content": doc,
                    "filename": results["metadatas"][i].get("filename", "unknown"),
                    "score": float(scores[top[i]]) if i < len(top) else 0.0,
                })
        return sources

    def hybrid_rrf(self, query: str, top_k: int = TOP_K) -> list[dict]:
        return self.engine.hybrid_search(query, top_k)

    def single_chunk_baseline(self, query: str, top_k: int = TOP_K) -> list[dict]:
        child_col = self.engine._get_child_collection()
        q_emb = self.engine.embedding_model.encode(query).tolist()
        results = child_col.query(query_embeddings=[q_emb], n_results=top_k)
        if not results or not results.get("documents") or not results["documents"][0]:
            return []
        sources = []
        for i in range(len(results["documents"][0])):
            sources.append({
                "content": results["documents"][0][i],
                "filename": results["metadatas"][0][i].get("filename", "unknown"),
                "score": 1.0 - (i * 0.05),
            })
        return sources


# ═══════════════════════════════════════════════════
# QA Pipeline
# ═══════════════════════════════════════════════════

def run_qa_pipeline(engine: RAGEngine, retriever, questions: list[str],
                    top_k: int = TOP_K) -> tuple[list[str], list[list[str]]]:
    """检索 -> 生成，返回 (answers, contexts_list)"""
    answers = []
    contexts_list = []

    for qi, q in enumerate(questions):
        sources = retriever(q, top_k)
        contexts = [s["content"] for s in sources]

        if not contexts:
            answers.append("（无相关文档）")
            contexts_list.append([])
            continue

        context_text = "\n\n---\n\n".join(contexts)
        today = datetime.now().strftime("%Y年%m月%d日")
        system_msg, user_msg = _build_prompt(context_text, q, today)

        try:
            resp = engine.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            answer = f"（LLM 调用失败: {e}）"

        answers.append(answer)
        contexts_list.append(contexts)
        print(f"  [{qi+1}/{len(questions)}] {q[:40]}... -> {len(sources)} sources, {len(answer)} chars")

    return answers, contexts_list


# ═══════════════════════════════════════════════════
# 消融实验
# ═══════════════════════════════════════════════════

def run_ablation(engine: RAGEngine, judge: LLMJudge,
                 cases: list[EvalCase]) -> list[AblationResult]:
    retrievers = BaselineRetrievers(engine)
    configs = [
        ("A. Vector Only (pure vector)", retrievers.vector_only),
        ("B. BM25 Only (keyword)", retrievers.bm25_only),
        ("C. Hybrid RRF (your design) **", retrievers.hybrid_rrf),
        ("D. Single-chunk (no Parent)", retrievers.single_chunk_baseline),
    ]

    results = []
    questions = [c.question for c in cases]

    for name, retriever in configs:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        t0 = time.time()
        answers, contexts_list = run_qa_pipeline(engine, retriever, questions)

        all_scores = []
        for i, case in enumerate(cases):
            scores = judge.score_all(
                case.question,
                answers[i] if i < len(answers) else "",
                contexts_list[i] if i < len(contexts_list) else [],
                case.ground_truth,
            )
            all_scores.append(scores)

        metrics = aggregate_scores(all_scores)
        elapsed = time.time() - t0

        result = AblationResult(name=name, metrics=metrics,
                                num_samples=len(cases), elapsed_sec=elapsed)
        results.append(result)
        print(f"  time={elapsed:.1f}s")
        for k, v in sorted(metrics.items()):
            bar = "█" * int(v * 20) if v >= 0 else "N/A"
            print(f"  {k:26s} = {v:.3f}  {bar}")

    return results


# ═══════════════════════════════════════════════════
# 快速回归模式 (--quick)
# ═══════════════════════════════════════════════════

QUICK_SEED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_seed_quick.json")
QUICK_METRICS = ["context_precision", "context_recall", "faithfulness"]

METRIC_CN = {
    "context_precision": "context_precision",
    "context_recall": "context_recall",
    "faithfulness": "faithfulness",
}


def run_quick_regression(engine: RAGEngine, judge: LLMJudge,
                         cases: list[EvalCase], top_k: int = TOP_K) -> tuple[list[dict], float]:
    """快速回归：5 条用例、混合检索、3 个核心指标。

    返回 (per_case_scores_list, elapsed_sec)
    """
    questions = [c.question for c in cases]

    t0 = time.time()
    answers, contexts_list = run_qa_pipeline(engine, engine.hybrid_search, questions, top_k)

    all_scores = []
    for i, case in enumerate(cases):
        scores = judge.score_quick(
            case.question,
            answers[i] if i < len(answers) else "",
            contexts_list[i] if i < len(contexts_list) else [],
            case.ground_truth,
        )
        all_scores.append(scores)

    elapsed = time.time() - t0
    return all_scores, elapsed


def load_baseline(path: str) -> dict | None:
    """加载基线 JSON 文件，返回 metrics 均值字典"""
    if not os.path.exists(path):
        print(f"  WARNING: Baseline file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "average" in data:
        return data["average"]
    return None


def format_quick_output(scores_list: list[dict], cases: list[EvalCase],
                        elapsed: float, baseline: dict | None = None) -> str:
    """格式化快速回归输出"""
    lines = []
    lines.append(f"\n=== Quick Regression ({len(cases)} cases, hybrid search only) ===\n")

    for i, (scores, case) in enumerate(zip(scores_list, cases)):
        lines.append(f"  [{i+1}] {case.question}")
        for metric in QUICK_METRICS:
            val = scores.get(metric, -1)
            if val >= 0:
                lines.append(f"       {metric}: {val:.2f}")
            else:
                lines.append(f"       {metric}: N/A")
        lines.append("")

    # 汇总平均值
    avg = aggregate_scores(scores_list)
    lines.append("  " + "─" * 45)
    lines.append(f"  Average ({len(cases)} cases):")
    for metric in QUICK_METRICS:
        val = avg.get(metric, -1)
        if val >= 0:
            if baseline and metric in baseline:
                diff = val - baseline[metric]
                sign = "+" if diff >= 0 else ""
                indicator = f"  ({sign}{diff:.2f})"
                lines.append(f"    {metric}: {val:.2f}{indicator}")
            else:
                lines.append(f"    {metric}: {val:.2f}")
    lines.append("  " + "─" * 45)
    lines.append(f"  Elapsed: {elapsed:.1f}s")
    lines.append(f"  Judge model: {LLM_MODEL}")

    return "\n".join(lines)


def save_quick_result(scores_list: list[dict], cases: list[EvalCase],
                      elapsed: float, output_path: str):
    """保存快速回归结果到 JSON"""
    avg = aggregate_scores(scores_list)
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": LLM_MODEL,
        "mode": "quick",
        "num_cases": len(cases),
        "elapsed_sec": round(elapsed, 1),
        "average": {k: round(v, 4) for k, v in avg.items()},
        "per_case": [
            {
                "question": case.question,
                "question_type": getattr(case, "question_type", ""),
                "metrics": {k: round(v, 4) for k, v in scores.items()},
            }
            for case, scores in zip(cases, scores_list)
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"\nQuick result saved: {output_path}")


# ═══════════════════════════════════════════════════
# 网格搜索模式 (--grid / --grid-file)
# ═══════════════════════════════════════════════════

GRID_OUTPUT_PATH = "eval_grid_result.json"


def parse_grid_args(grid_args: list[str]) -> dict[str, list]:
    """解析 --grid 参数，如 'BM25_WEIGHT:0.2,0.3,0.4' → {'BM25_WEIGHT': [0.2, 0.3, 0.4]}"""
    result = {}
    for arg in grid_args:
        if ":" not in arg:
            print(f"  WARNING: Invalid grid format '{arg}', expected 'NAME:val1,val2,...'")
            continue
        name, vals_str = arg.split(":", 1)
        name = name.strip()
        try:
            vals = [float(v.strip()) for v in vals_str.split(",")]
        except ValueError:
            vals = [v.strip() for v in vals_str.split(",")]
        result[name] = vals
    return result


def load_grid_file(path: str) -> dict[str, list]:
    """从 JSON 文件加载网格参数"""
    if not os.path.exists(path):
        print(f"ERROR: Grid file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = data.get("params", data)
    return params


def _grid_cartesian_product(params: dict[str, list]) -> list[dict]:
    """计算参数的笛卡尔积"""
    if not params:
        return [{}]
    keys = list(params.keys())
    values = list(params.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def _get_current_config_values() -> dict:
    """读取 config.py 中的当前参数值"""
    return {
        "BM25_WEIGHT": config_module.BM25_WEIGHT,
        "VECTOR_WEIGHT": config_module.VECTOR_WEIGHT,
        "K_RRF": config_module.K_RRF,
    }


def _apply_config_overrides(params: dict) -> dict:
    """临时修改 config 模块和 rag_engine 模块中的参数值。

    返回原始值字典，用于后续恢复。
    """
    originals = {}
    for name, value in params.items():
        # 保存原始值
        if hasattr(config_module, name):
            originals[name] = getattr(config_module, name)
        elif hasattr(rag_engine_module, name):
            originals[name] = getattr(rag_engine_module, name)
        else:
            print(f"  WARNING: Unknown config param '{name}', skipping")
            continue

        # 设置新值到 config 模块
        if hasattr(config_module, name):
            setattr(config_module, name, value)
        # 也设置到 rag_engine 模块（因为 from import 创建了独立绑定）
        if hasattr(rag_engine_module, name):
            setattr(rag_engine_module, name, value)

    return originals


def _restore_config_overrides(originals: dict):
    """恢复 config 模块和 rag_engine 模块的原始值"""
    for name, value in originals.items():
        if hasattr(config_module, name):
            setattr(config_module, name, value)
        if hasattr(rag_engine_module, name):
            setattr(rag_engine_module, name, value)


def _format_config_label(params: dict) -> str:
    """格式化配置标签，如 'BM25=0.3, VEC=0.7, k=60'"""
    parts = []
    for name, value in params.items():
        short = name.replace("_WEIGHT", "").replace("K_", "k").replace("_", "")
        if isinstance(value, float) and value == int(value):
            value = int(value)
        parts.append(f"{short}={value}")
    return ", ".join(parts)


def run_grid_search(engine: RAGEngine, judge: LLMJudge,
                    cases: list[EvalCase], grid_params: dict[str, list],
                    top_k: int = TOP_K) -> list[dict]:
    """网格搜索：遍历参数组合，每组跑一次快速回归。

    返回 list of dict，每个包含 config_label, metrics, elapsed_sec
    """
    combos = _grid_cartesian_product(grid_params)
    print(f"\nGrid search: {len(combos)} configs = {list(grid_params.keys())}")
    print(f"Each config runs {len(cases)} quick cases\n")

    current_config = _get_current_config_values()
    results = []

    for ci, combo in enumerate(combos):
        label = _format_config_label(combo)
        is_current = all(
            abs(combo.get(k, 0) - current_config.get(k, 0)) < 1e-9
            for k in combo
        )

        # 打印当前配置
        current_marker = " (current)" if is_current else ""
        print(f"[{ci+1}/{len(combos)}] {label}{current_marker}")

        # 应用配置覆盖
        originals = _apply_config_overrides(combo)

        try:
            scores_list, elapsed = run_quick_regression(engine, judge, cases, top_k)
            avg = aggregate_scores(scores_list)

            result = {
                "config_label": label,
                "params": dict(combo),
                "metrics": {k: round(v, 4) for k, v in avg.items()},
                "elapsed_sec": round(elapsed, 1),
                "is_current": is_current,
            }
            results.append(result)

            # 简要输出
            p = avg.get("context_precision", -1)
            r = avg.get("context_recall", -1)
            f = avg.get("faithfulness", -1)
            print(f"       precision={p:.3f}  recall={r:.3f}  faith={f:.3f}  time={elapsed:.1f}s")
        finally:
            # 恢复原始值
            _restore_config_overrides(originals)

    print(f"\nConfig values restored to original.")

    # 打印对比表
    print(format_grid_table(results))
    return results


def format_grid_table(grid_results: list[dict]) -> str:
    """格式化网格搜索结果对比表"""
    if not grid_results:
        return "（无结果）"

    lines = [""]
    lines.append(f"=== Grid Search Results ({len(grid_results)} configs) ===\n")

    # 表头
    header = f"  {'Config':<42s} {'precision':>9s}  {'recall':>9s}  {'faith':>9s}  {'time':>6s}"
    lines.append(header)
    lines.append("  " + "─" * 82)

    # Find best in each column
    best = {}
    for metric in QUICK_METRICS:
        vals = [(r["metrics"].get(metric, -1), i) for i, r in enumerate(grid_results)
                if r["metrics"].get(metric, -1) >= 0]
        if vals:
            best[metric] = max(vals)[0]

    for r in grid_results:
        label = r["config_label"]
        current_marker = " *" if r.get("is_current") else ""
        m = r["metrics"]
        p_str = f"{m.get('context_precision', -1):.2f}" if m.get('context_precision', -1) >= 0 else "N/A"
        r_str = f"{m.get('context_recall', -1):.2f}" if m.get('context_recall', -1) >= 0 else "N/A"
        f_str = f"{m.get('faithfulness', -1):.2f}" if m.get('faithfulness', -1) >= 0 else "N/A"

        p_best = " **" if abs(m.get('context_precision', -1) - best.get('context_precision', -1)) < 0.001 else ""
        r_best = " **" if abs(m.get('context_recall', -1) - best.get('context_recall', -1)) < 0.001 else ""
        f_best = " **" if abs(m.get('faithfulness', -1) - best.get('faithfulness', -1)) < 0.001 else ""

        p_str += p_best
        r_str += r_best
        f_str += f_best

        row = f"  {label+current_marker:<42s} {p_str:>9s}  {r_str:>9s}  {f_str:>9s}  {r['elapsed_sec']:.1f}s"
        lines.append(row)

    lines.append("  " + "─" * 82)
    lines.append("  * = current config.py value")
    lines.append("  ** = best in column")
    return "\n".join(lines)


def save_grid_result(grid_results: list[dict], output_path: str):
    """保存网格搜索结果到 JSON"""
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": LLM_MODEL,
        "mode": "grid",
        "num_configs": len(grid_results),
        "results": grid_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"Grid result saved: {output_path}")


# ═══════════════════════════════════════════════════
# 断点恢复 (--resume / 隐式)
# ═══════════════════════════════════════════════════

def check_resume(output_path: str) -> tuple[list[str], dict | None]:
    """检查是否有可恢复的结果。

    返回 (completed_configs, existing_data_or_None)
    """
    if not os.path.exists(output_path):
        return [], None

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_results = data.get("results", [])
    completed = [r.get("config", r.get("config_label", "")) for r in existing_results]

    return completed, data


def resume_ablation(engine: RAGEngine, judge: LLMJudge,
                    cases: list[EvalCase], output_path: str) -> list[AblationResult]:
    """支持断点恢复的消融实验"""
    all_configs = [
        "A. Vector Only (pure vector)",
        "B. BM25 Only (keyword)",
        "C. Hybrid RRF (your design) **",
        "D. Single-chunk (no Parent)",
    ]

    completed_names, existing_data = check_resume(output_path)

    if completed_names:
        skipped = [c for c in completed_names if any(ac in c for ac in all_configs)]
        if skipped:
            print(f"\n=== Resume detected ===")
            print(f"  Skipped {len(skipped)} already-completed configs: {skipped}")

    # 找出未完成的配置
    remaining_configs = []
    for ac in all_configs:
        if not any(ac in c for c in completed_names):
            remaining_configs.append(ac)

    if not remaining_configs:
        print("  All configs already completed!")
        # 返回已有结果
        retrievers = BaselineRetrievers(engine)
        results = []
        for r in existing_data.get("results", []):
            results.append(AblationResult(
                name=r["config"],
                metrics=r.get("metrics", {}),
                num_samples=existing_data.get("num_test_cases", 0),
                elapsed_sec=r.get("elapsed_sec", 0),
            ))
        return results

    # 只跑未完成的配置
    retrievers = BaselineRetrievers(engine)
    config_map = {
        "A. Vector Only (pure vector)": retrievers.vector_only,
        "B. BM25 Only (keyword)": retrievers.bm25_only,
        "C. Hybrid RRF (your design) **": retrievers.hybrid_rrf,
        "D. Single-chunk (no Parent)": retrievers.single_chunk_baseline,
    }

    questions = [c.question for c in cases]
    new_results = []

    for name in remaining_configs:
        retriever = config_map[name]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        t0 = time.time()
        answers, contexts_list = run_qa_pipeline(engine, retriever, questions)

        all_scores = []
        for i, case in enumerate(cases):
            scores = judge.score_all(
                case.question,
                answers[i] if i < len(answers) else "",
                contexts_list[i] if i < len(contexts_list) else [],
                case.ground_truth,
            )
            all_scores.append(scores)

        metrics = aggregate_scores(all_scores)
        elapsed = time.time() - t0

        result = AblationResult(name=name, metrics=metrics,
                                num_samples=len(cases), elapsed_sec=elapsed)
        new_results.append(result)
        print(f"  time={elapsed:.1f}s")
        for k, v in sorted(metrics.items()):
            bar = "█" * int(v * 20) if v >= 0 else "N/A"
            print(f"  {k:26s} = {v:.3f}  {bar}")

    # 合并已有结果和新结果
    if existing_data:
        existing_abl_results = []
        for r in existing_data.get("results", []):
            existing_abl_results.append(AblationResult(
                name=r["config"],
                metrics=r.get("metrics", {}),
                num_samples=existing_data.get("num_test_cases", 0),
                elapsed_sec=r.get("elapsed_sec", 0),
            ))
        # 去重：用 config name 去重
        seen = {r.name: r for r in existing_abl_results}
        for nr in new_results:
            seen[nr.name] = nr
        return list(seen.values())

    return new_results


# ═══════════════════════════════════════════════════
# 结果格式化
# ═══════════════════════════════════════════════════

METRIC_LABELS = {
    "context_precision": "c_precision",
    "context_recall": "c_recall",
    "faithfulness": "faithfulness",
    "answer_relevancy": "a_relevancy",
    "answer_correctness": "a_correctness",
    "context_entity_recall": "c_entity_rec",
}


def format_results_table(results: list[AblationResult]) -> str:
    if not results:
        return "（无结果）"

    all_metrics = sorted(set().union(*(r.metrics.keys() for r in results)))
    best = {}
    for m in all_metrics:
        vals = [r.metrics.get(m, 0) for r in results if m in r.metrics and r.metrics.get(m, -1) >= 0]
        if vals:
            best[m] = max(vals)

    lines = [""]
    lines.append("╔" + "═" * 78 + "╗")
    lines.append("║  Ablation Study -- RAG Retrieval Strategy Comparison" + " " * 28 + "║")
    lines.append("╟" + "═" * 28 + "╤" + "═" * 50 + "╡")

    header = "║ {:<26s} │".format("Config")
    for m in all_metrics:
        header += " {:^11s} │".format(METRIC_LABELS.get(m, m[:12]))
    lines.append(header)
    lines.append("╟" + "─" * 28 + "┼" + "─" * 50 + "╡")

    for r in results:
        name = r.name.split("(")[0].strip()[:26]
        row = "║ {:<26s} │".format(name)
        for m in all_metrics:
            val = r.metrics.get(m)
            if val is not None and val >= 0:
                marker = " *" if abs(val - best.get(m, 0)) < 0.001 else ""
                row += " {:>8.3f}{:<2s} │".format(val, marker)
            else:
                row += " {:>10s} │".format("N/A")
        lines.append(row)

    lines.append("╚" + "═" * 28 + "╧" + "═" * 50 + "╝")
    lines.append("  * = best in column")
    return "\n".join(lines)


def save_results(results: list[AblationResult], output_path: str):
    # CSV
    rows = []
    for r in results:
        row = {"config": r.name.split("(")[0].strip(), "samples": r.num_samples,
               "elapsed_sec": round(r.elapsed_sec, 1)}
        row.update({k: round(v, 4) for k, v in r.metrics.items() if v >= 0})
        rows.append(row)

    csv_path = output_path.replace(".json", ".csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        if rows:
            f.write(",".join(rows[0].keys()) + "\n")
            for row in rows:
                f.write(",".join(str(v) for v in row.values()) + "\n")
    print(f"\nCSV saved: {csv_path}")

    # JSON
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": LLM_MODEL,
        "num_test_cases": results[0].num_samples if results else 0,
        "results": [
            {"config": r.name, "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
             "elapsed_sec": round(r.elapsed_sec, 1)}
            for r in results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {output_path}")


# ═══════════════════════════════════════════════════
# 测试集加载
# ═══════════════════════════════════════════════════

def load_testset(path: str, limit: int = 0) -> list[EvalCase]:
    """从 JSON 文件加载测试集"""
    if not os.path.exists(path):
        print(f"ERROR: Testset not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if limit > 0:
        test_data = test_data[:limit]

    cases = []
    for item in test_data:
        case = EvalCase(
            question=item.get("question", ""),
            ground_truth=item.get("ground_truth", ""),
            reference_contexts=item.get("reference_contexts", []),
        )
        # 附加 question_type 用于输出
        case.question_type = item.get("question_type", "")
        cases.append(case)

    return cases


# ═══════════════════════════════════════════════════
# RAG 引擎初始化（含种子文档）
# ═══════════════════════════════════════════════════

def init_engine_with_seed() -> RAGEngine:
    """初始化 RAG 引擎，如果 ChromaDB 为空则写入种子文档"""
    print("Initializing RAG engine...")
    engine = RAGEngine()

    child_col = engine._get_child_collection()
    try:
        count = child_col.count()
    except Exception:
        count = 0

    if count == 0:
        print("ChromaDB is empty -- seeding test document...")
        seed_doc = """# Company Org Structure

## Departments
The company has three core departments: R&D, Marketing, Finance.

## R&D Dept
R&D lead is Zhang San, with frontend team (12 people) and backend team (13 people).
R&D total: 25 people.

## Marketing Dept
Marketing lead is Li Si, responsible for brand promotion and market research.
Marketing has 10 employees, distributed in Beijing and Shanghai.

## Finance Dept
Finance lead is Wang Wu, CPA. Responsible for budgeting and financial reporting.
Finance has 5 people, office in Beijing HQ.

## Company Overview
Founded March 2020, HQ in Haidian District, Beijing.
2024 annual revenue: 50 million RMB, YoY growth 30%.

## Culture
"Tech-driven, customer-first" values.
Free lunch and gym membership for employees.
"""
        engine.add_document(99999, "_eval_seed.md", seed_doc)
        print("  Seeded 1 document with company org info")

    return engine


def init_judge() -> tuple[OpenAI, LLMJudge]:
    """初始化 LLM Judge"""
    if not LLM_API_KEY:
        print("ERROR: LLM_API_KEY not configured in .env")
        sys.exit(1)
    llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    judge = LLMJudge(llm)
    print(f"LLM Judge: {LLM_MODEL} @ {LLM_API_BASE}")
    return llm, judge


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAGAS-standard 6-metric evaluation + ablation study + quick/grid/resume modes")
    parser.add_argument("--testset", help="Path to eval testset JSON")
    parser.add_argument("--output", default="eval_results.json", help="Output path for results")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation, only evaluate hybrid search")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate first N cases (for quick testing)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of retrieved chunks (default: {TOP_K})")
    # ── 新增参数 ──
    parser.add_argument("--quick", action="store_true",
                        help="Quick regression mode: 5 seed cases, hybrid only, 3 core metrics")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline quick result JSON for comparison")
    parser.add_argument("--grid", nargs="*", default=None,
                        help='Grid search: "PARAM:val1,val2,..." (multiple allowed)')
    parser.add_argument("--grid-file", type=str, default=None,
                        help="Path to grid params JSON file")
    args = parser.parse_args()

    # ── 检查互斥 ──
    if args.quick and args.testset:
        print("WARNING: --quick uses built-in seed testset, --testset will be ignored for quick mode")

    # ── 模式路由 ──

    # 网格搜索模式
    if args.grid or args.grid_file:
        return _main_grid(args)

    # 快速回归模式
    if args.quick:
        return _main_quick(args)

    # 标准评估模式（含隐式断点恢复）
    return _main_standard(args)


def _main_quick(args):
    """快速回归模式入口"""
    # 1. Load seed testset
    if not os.path.exists(QUICK_SEED_PATH):
        print(f"ERROR: Quick seed testset not found: {QUICK_SEED_PATH}")
        print(f"       Expected 5 cases in eval_seed_quick.json")
        sys.exit(1)

    cases = load_testset(QUICK_SEED_PATH)
    print(f"Loaded {len(cases)} quick seed cases")

    # 2. Init engine
    engine = init_engine_with_seed()

    # 3. Init judge
    _, judge = init_judge()

    # 4. Run quick regression
    scores_list, elapsed = run_quick_regression(engine, judge, cases, args.top_k)

    # 5. Load baseline if provided
    baseline = None
    if args.baseline:
        print(f"\nLoading baseline: {args.baseline}")
        baseline = load_baseline(args.baseline)

    # 6. Format output
    print(format_quick_output(scores_list, cases, elapsed, baseline))

    # 7. Save result
    save_quick_result(scores_list, cases, elapsed, "eval_quick_result.json")


def _main_grid(args):
    """网格搜索模式入口"""
    # 1. Parse grid params
    if args.grid_file:
        grid_params = load_grid_file(args.grid_file)
        print(f"Loaded grid params from {args.grid_file}: {grid_params}")
    elif args.grid:
        if not args.grid:
            print("ERROR: --grid requires at least one parameter, e.g. --grid \"BM25_WEIGHT:0.2,0.3\"")
            sys.exit(1)
        grid_params = parse_grid_args(args.grid)
        print(f"Parsed grid params: {grid_params}")
    else:
        print("ERROR: Must specify --grid or --grid-file")
        sys.exit(1)

    if not grid_params:
        print("ERROR: No valid grid parameters found")
        sys.exit(1)

    # 2. Load seed testset
    if not os.path.exists(QUICK_SEED_PATH):
        print(f"ERROR: Quick seed testset not found: {QUICK_SEED_PATH}")
        sys.exit(1)

    cases = load_testset(QUICK_SEED_PATH)
    print(f"Loaded {len(cases)} quick seed cases")

    # 3. Init engine
    engine = init_engine_with_seed()

    # 4. Init judge
    _, judge = init_judge()

    # 5. Run grid search
    grid_results = run_grid_search(engine, judge, cases, grid_params, args.top_k)

    # 6. Save
    save_grid_result(grid_results, GRID_OUTPUT_PATH)


def _main_standard(args):
    """标准评估模式入口（含隐式断点恢复）"""
    # ── 1. Load testset ──
    if not args.testset:
        print("ERROR: --testset is required (unless using --quick or --grid)")
        print(f"       Usage: uv run python eval_ragas.py --testset eval_testset_sample.json")
        sys.exit(1)

    cases = load_testset(args.testset, args.limit)
    print(f"Loaded {len(cases)} test cases")

    # ── 2. Init RAG engine ──
    engine = init_engine_with_seed()

    # ── 3. Init LLM Judge ──
    _, judge = init_judge()

    # ── 4. Run ──
    if args.skip_ablation:
        print(f"\n{'='*60}")
        print("  Single Evaluation -- Hybrid Search (your design)")
        print(f"{'='*60}")
        questions = [c.question for c in cases]
        t0 = time.time()
        answers, contexts_list = run_qa_pipeline(engine, engine.hybrid_search, questions, args.top_k)

        all_scores = []
        for i, case in enumerate(cases):
            scores = judge.score_all(case.question, answers[i], contexts_list[i], case.ground_truth)
            all_scores.append(scores)

        metrics = aggregate_scores(all_scores)
        elapsed = time.time() - t0
        results = [AblationResult(
            name="C. Hybrid RRF (your design) **",
            metrics=metrics, num_samples=len(cases), elapsed_sec=elapsed,
        )]
    else:
        # 检查是否可以断点恢复
        if os.path.exists(args.output):
            results = resume_ablation(engine, judge, cases, args.output)
        else:
            results = run_ablation(engine, judge, cases)

    # ── 5. Output ──
    print(format_results_table(results))

    # ── 6. Summary for resume ──
    best_correctness = max(results, key=lambda r: r.metrics.get("answer_correctness", 0))
    best_recall = max(results, key=lambda r: r.metrics.get("context_recall", 0))
    best_faith = max(results, key=lambda r: r.metrics.get("faithfulness", 0))

    print(f"\n--- Resume-ready summary ---")
    print(f"Evaluated on {len(cases)} multi-scenario test cases with {LLM_MODEL} as judge:")
    print(f"  context_recall:    {best_recall.metrics.get('context_recall', 0):.1%}")
    print(f"  faithfulness:      {best_faith.metrics.get('faithfulness', 0):.1%}")
    print(f"  answer_correctness:{best_correctness.metrics.get('answer_correctness', 0):.1%}")

    # ── 7. Save ──
    save_results(results, args.output)

    # ── 8. Metric interpretation ──
    print(f"""
{'='*60}
Metric interpretation
{'='*60}
context_precision   -- retrieval precision. >0.7 good, <0.5 noisy
context_recall      -- retrieval coverage. >0.8 good, <0.6 increase top_k
faithfulness        -- generation faithfulness. >0.8 good, <0.5 hallucinating
answer_relevancy    -- answer relevance. >0.8 good, <0.5 off-topic
answer_correctness  -- answer correctness. >0.7 good
context_entity_recall -- entity recall. maps to your "keyword hit rate"
{'='*60}
""")


if __name__ == "__main__":
    main()

"""
手写 ReAct Agent — 求职面试官

架构：
  start_interview: → ReAct(analyze_profile → generate_question → final)
  answer_question: → ReAct(score_answer → [generate_question] → final)
  generate_report: → ReAct(generate_report → final)

  每次用户操作触发一轮完整的 ReAct 循环（Thought → Action → Tool → Observation → ... → Final Answer）
"""
import json
import logging

from openai import OpenAI

from config import LLM_MODEL

logger = logging.getLogger(__name__)


def _parse_llm_json(raw: str | None) -> dict:
    """解析 LLM 返回的 JSON，自动剥离 ```json ``` markdown 包裹"""
    content = (raw or "{}").strip()
    if not content:
        return {}
    # 剥离 markdown 代码块
    if content.startswith("```"):
        lines = content.split("\n", 1)
        content = lines[1] if len(lines) > 1 else content[3:]
        # 去掉末尾的 ```
        if "```" in content:
            content = content.rsplit("```", 1)[0]
        content = content.strip()
    return json.loads(content)

# ────────────────────── 共享枚举常量 ──────────────────────
# 以下常量被 TOOL_SCHEMAS 和 REACT_TOOLS 共同引用，避免字符串重复

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]       # 难度分级
SCORE_DIMENSIONS = ["技术深度", "表达清晰度", "逻辑性"]  # 评分维度


# ────────────────────── Prompt 模板 ──────────────────────


def _profile_prompt() -> str:
    return """你是一个资深的 HR 技术面试官。请分析以下简历内容，输出 JSON：
{
  "skills": ["Python", "FastAPI", "..."],
  "experience_years": 0,
  "education": "本科",
  "gaps": ["缺少系统设计经验"],
  "match_score": 0.75,
  "recommended_dimensions": ["技术深度", "项目经验", "系统设计", "行为面试"]
}"""


def _question_prompt(resume_content: str, jd_content: str,
                     dimension: str, difficulty: str, history: str,
                     avoid_questions: list | None = None) -> str:
    ctx = f"简历内容：\n{resume_content[:2000]}\n"
    if jd_content:
        ctx += f"\n岗位要求：\n{jd_content[:1000]}\n"
    prompt = (
        f"你是一个技术面试官。根据以下候选人信息和面试进度出题。\n\n"
        f"{ctx}\n"
        f"本题建议维度：{dimension}  建议难度：{difficulty}\n"
        f"（注：建议值仅供参考，你可以根据实际情况调整维度和难度）\n\n"
        f"本场已答题目：\n{history}\n\n"
        f"规则：\n"
        f"1. 【基于项目】从候选人实际项目经验出发，不问简历上没有的技术\n"
        f"2. 【针对 JD】如有岗位描述，针对 JD 要求的能力提问\n"
        f"3. 【区分维度】不要连续问相同维度的问题，覆盖面要广\n"
        f"4. 【避免重复】题目内容不得与已答题目重复\n"
        f"5. 【友好提问】只出能用自然语言回答的题，不要出写代码、写伪代码或写 SQL 的题\n"
    )
    if avoid_questions:
        prompt += (
            f"\n⚠️ 以下题目是之前面试中问过的，本次必须完全避开：\n"
            + "\n".join(f"- {q}" for q in avoid_questions)
            + "\n"
        )
    prompt += (
        f"\n输出 JSON：\n"
        f'{{"question": "请描述...", "dimension": "技术深度/项目经验/系统设计/行为面试", '
        f'"difficulty": "easy/medium/hard", "expected_keywords": ["关键词"], '
        f'"hint": "回答建议"}}'
    )
    return prompt


def _score_prompt(question: str, dimension: str, keywords: str, answer: str) -> str:
    return f"""请对以下回答进行多维度评分（每项 0-10）。

问题：{question}
维度：{dimension}
期望关键词：{keywords}
回答：{answer}

评分标准：
- 技术深度：是否理解原理而非仅用过
- 表达清晰度：语言通顺、人类能听懂即可，不要求专业术语，不打比方举例子也没问题
- 逻辑性：前后是否一致、自洽

⚠️ 反作弊规则（严格执行）：
- 如果回答直接复制/重复了题目原文 → 总分直接给 0
- 如果回答答非所问、敷衍了事 → 总分直接给 0
- 只有真正展示了理解、经验和思考的回答才能得到 5 分以上

输出 JSON（评分范围 0-10，0 分合法）：
{{"total_score": 0, "dimensions": {{"技术深度": 0, "表达清晰度": 0, "逻辑性": 0}}, "feedback": "无效回答", "strengths": [], "improvements": []}}"""


def _report_prompt(qa_history: str) -> str:
    return f"""请根据以下面试记录生成详细报告。

{qa_history}

输出 JSON（严格按此格式）：
{{
  "overall_score": 7.2,
  "dimension_scores": {{"技术深度": 7.5, "表达清晰度": 6.8, "逻辑性": 8.0}},
  "question_analysis": [
    {{"number": 1, "score": 8.0, "feedback_summary": "技术原理理解充分", "improvement_suggestion": "可以补充具体数据"}},
    {{"number": 2, "score": 6.0, "feedback_summary": "表达不够结构化", "improvement_suggestion": "建议用 STAR 原则"}}
  ],
  "keyword_analysis": {{
    "strong_keywords": ["Python", "FastAPI", "微服务"],
    "missed_keywords": ["Docker", "Redis"],
    "hit_rate": 0.65
  }},
  "difficulty_distribution": {{"easy": 0, "medium": 5, "hard": 2}},
  "strengths": "技术基础扎实，逻辑清晰",
  "weaknesses": "系统设计经验尚浅，表达可更结构化",
  "learning_suggestions": ["深入学习系统设计", "练习 STAR 表达"],
  "summary": "整体表现中等偏上，技术基础过关"
}}

分析要求：
1. question_analysis：对每一题单独给出评分摘要和改进建议
2. keyword_analysis：对比期望关键词，给出命中率和遗漏项
3. difficulty_distribution：统计各难度题目数量
4. strengths/weaknesses：基于实际题目表现总结，不要空泛
5. learning_suggestions：可操作的具体学习建议"""


# ────────────────────── Agent ──────────────────────


class InterviewAgent:
    """手写 ReAct Agent — 每次用户操作触发一轮完整 ReAct 循环"""

    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    # ────────────────────── Tool 实现（各 _exec_* 直接调 LLM + 解析 JSON） ──────────────────────

    def _exec_analyze(self, resume: str, jd: str) -> dict:
        body = "=== 简历 ===\n" + resume[:2000] + "\n\n"
        if jd:
            body += "=== 岗位描述 ===\n" + jd[:1000] + "\n\n"
        body += "请分析。"
        resp = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": _profile_prompt()},
                      {"role": "user", "content": body}],
            temperature=0.1, max_tokens=4096,
        )
        return _parse_llm_json(resp.choices[0].message.content)

    def _exec_question(self, dimension: str, difficulty: str,
                       history: str, resume: str, jd: str,
                       avoid_questions: list | None = None) -> dict:
        resp = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "你是一个技术面试官，直接输出 JSON。"},
                      {"role": "user", "content": _question_prompt(resume, jd, dimension, difficulty, history, avoid_questions)}],
            temperature=0.3, max_tokens=4096,
        )
        return _parse_llm_json(resp.choices[0].message.content)

    def _exec_score(self, question: str, dimension: str,
                    keywords: str, answer: str) -> dict:
        resp = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "你是一个面试评分官，直接输出 JSON。"},
                      {"role": "user", "content": _score_prompt(question, dimension, keywords, answer)}],
            temperature=0, max_tokens=4096,
        )
        return _parse_llm_json(resp.choices[0].message.content)

    def _exec_report(self, qa_history: str) -> dict:
        resp = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "你是一个面试总结官，直接输出 JSON。"},
                      {"role": "user", "content": _report_prompt(qa_history)}],
            temperature=0.2, max_tokens=4096,
        )
        return _parse_llm_json(resp.choices[0].message.content)

    # ────────────────────── 对外接口（确定性顺序，无 ReAct 循环） ──────────────────────

    def start_interview(self, resume_content: str, jd_content: str = "",
                        total_questions: int = 8, avoid_questions: list | None = None) -> dict:
        """开始面试：顺序调 analyze_profile → generate_question，共 2 次 LLM 调用"""
        logger.info("Agent 开始面试分析...")

        profile = self._exec_analyze(resume_content, jd_content)
        first_q = self._exec_question(
            "技术深度", "medium", "[]",
            resume_content, jd_content,
            avoid_questions,
        )

        logger.info("Profile: skills=%s", profile.get("skills", [])[:3])
        return {"profile": profile, "first_question": first_q}

    def answer_question(self, question_text: str, dimension: str,
                        difficulty: str, expected_keywords: list,
                        user_answer: str,
                        question_number: int, total_questions: int,
                        history: list | None = None,
                        resume_content: str = "", jd_content: str = "",
                        avoid_questions: list | None = None) -> dict:
        """提交回答：顺序调 score_answer → [generate_question]，共 1~2 次 LLM 调用"""
        logger.info("Agent 评分第 %d 题...", question_number)

        kw_json = json.dumps(expected_keywords or [], ensure_ascii=False)
        score = self._exec_score(question_text, dimension, kw_json, user_answer)

        next_q = None
        is_complete = question_number >= total_questions
        if not is_complete:
            history_json = json.dumps(history or [], ensure_ascii=False)
            next_q = self._exec_question(
                dimension, difficulty, history_json,
                resume_content, jd_content,
                avoid_questions,
            )

        return {"score": score, "next_question": next_q, "is_complete": is_complete}

    def generate_report(self, qa_history: list) -> dict:
        """生成面试报告：直接调 _exec_report，共 1 次 LLM 调用"""
        logger.info("Agent 生成面试报告...")
        try:
            qa_json = json.dumps(qa_history or [], ensure_ascii=False)
            return self._exec_report(qa_json)
        except Exception as e:
            logger.error("生成报告失败: %s", e)
            return {"overall_score": 0, "strengths": "", "weaknesses": "",
                    "summary": "报告生成失败", "learning_suggestions": [],
                    "dimension_scores": {}, "question_analysis": [],
                    "keyword_analysis": {}, "difficulty_distribution": {}}

"""
FastAPI 主应用 — 路由定义 + 启动逻辑
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import hashlib, json, asyncio, hmac, time, secrets
import logging

from config import CHROMA_PERSIST_DIR, APP_SECRET_KEY, MAX_UPLOAD_SIZE
from schemas import (
    UploadResponse, AskRequest, AskResponse,
    SourceItem, DocumentItem,
    InterviewStartRequest, InterviewAnswerRequest,
)
from database import (
    async_init_db, async_save_document, async_get_documents,
    async_delete_document, async_update_chunk_count,
    async_get_document_by_hash, async_get_document_content,
    async_create_interview_session, async_get_interview_session,
    async_update_interview_session, async_add_interview_question,
    async_update_interview_answer, async_get_interview_questions,
    async_save_interview_report, async_get_interview_report,
    async_get_previous_questions_by_resume,
)
from rag_engine import RAGEngine
from pdf_handler import extract_text
from interview_agent import InterviewAgent


logger = logging.getLogger(__name__)

# ────────────────────── 全局变量 ──────────────────────
# 注意：RAGEngine 内部有 sentence-transformers 模型（~80MB）
# 所以在应用启动时只加载一次，所有请求共享
rag_engine: RAGEngine = None
interview_agent: InterviewAgent = None


# ────────────────────── 生命周期管理 ──────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 的启动/关闭钩子：
    - 启动时：初始化 SQLite 表 + 加载 RAG 引擎
    - 关闭时：ChromaDB 自动持久化，无需手动操作
    """
    global rag_engine
    logger.info("正在启动 RAG 引擎...")

    # 1. 初始化 SQLite 表
    await async_init_db()

    # 2. 初始化 RAG 引擎（加载模型 + 连接 ChromaDB）
    rag_engine = RAGEngine()
    logger.info("RAG 引擎启动完成")

    # 3. 初始化面试 Agent（复用 RAG 引擎的 LLM 客户端）
    global interview_agent
    interview_agent = InterviewAgent(rag_engine.llm)
    logger.info("面试 Agent 启动完成")

    yield

    logger.info("正在关闭 RAG 引擎...")


# ────────────────────── 创建应用 ──────────────────────

app = FastAPI(
    title="Interview Engine",
    description="AI-Powered Mock Interview System — RAG-based resume analysis + ReAct Agent-driven interview simulation",
    version="1.0.0",
    lifespan=lifespan
)

# ────────────────────── CORS（跨域白名单） ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8765", "http://localhost:8765"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ────────────────────── 结构化日志中间件（注入 request_id） ──────────────────────

import uuid

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """为每个请求注入 request_id，便于日志追踪"""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ────────────────────── 健康检查端点 ──────────────────────

@app.get("/health")
async def health_check():
    """容器编排/负载均衡健康检查"""
    return {"status": "ok", "service": "interview-engine", "version": "1.0.0"}

# ────────────────────── 安全响应头 ──────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ────────────────────── 安全：API Token 认证 ──────────────────────
# 策略：
# 1. 外部 API 客户端：用 APP_SECRET_KEY 作为 Bearer token（长期）
# 2. Web 前端：通过 /auth/session 获取短期 session token（1 小时，HMAC 签名）
# 3. 未配置 APP_SECRET_KEY → 启动时打印警告，但所有 API fail closed
# 4. Token 不注入 HTML，前端在运行时通过 /auth/session 获取

SESSION_TOKEN_EXPIRE = 3600  # 1 小时

_security = HTTPBearer(auto_error=True)


def _issue_session_token() -> str:
    """用 HMAC-SHA256 签发短期 session token"""
    if not APP_SECRET_KEY:
        raise HTTPException(500, "认证未配置，服务器无法启动")
    expiry = int(time.time()) + SESSION_TOKEN_EXPIRE
    payload = f"session:{expiry}"
    sig = hmac.new(APP_SECRET_KEY.encode(), payload.encode(), "sha256").hexdigest()[:16]
    return f"{payload}:{sig}"


def _validate_session_token(token: str) -> bool:
    """验证 session token 签名和有效期"""
    parts = token.split(":")
    if len(parts) != 3 or parts[0] != "session":
        return False
    _, expiry_str, sig = parts
    try:
        expiry = int(expiry_str)
    except ValueError:
        return False
    if time.time() > expiry:
        return False
    payload = f"session:{expiry}"
    expected = hmac.new(APP_SECRET_KEY.encode(), payload.encode(), "sha256").hexdigest()[:16]
    return hmac.compare_digest(sig, expected)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(_security)):
    """
    验证 Bearer token：
    - 匹配 APP_SECRET_KEY → 外部 API 调用
    - 匹配 session token → Web 前端调用
    - 都不匹配 → 401
    """
    if not APP_SECRET_KEY:
        raise HTTPException(
            500, "服务器认证未配置。请在 .env 中设置 APP_SECRET_KEY。"
        )
    token = credentials.credentials
    if token == APP_SECRET_KEY:
        return  # 外部 API 客户端
    if _validate_session_token(token):
        return  # Web 前端 session
    raise HTTPException(401, "无效或过期的认证令牌")


@app.post("/auth/session")
async def auth_session_token():
    """
    Web 前端获取 session token（短期，1 小时有效）
    返回：{"token": "session:1234567890:abc123"}
    前端在运行时存储此 token（不在 HTML 源码中），用于后续 API 调用
    """
    if not APP_SECRET_KEY:
        raise HTTPException(500, "服务器认证未配置")
    token = _issue_session_token()
    return {"token": token, "expires_in": SESSION_TOKEN_EXPIRE}


# ────────────────────── API 路由 ──────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """智能知识库问答 — Web 界面（从 templates/index.html 加载）"""
    import os
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = Form('general'),
    _=Depends(verify_token)
):
    """
    上传文档
    - 支持 .txt、.md、.pdf 和 .docx
    - 自动切分 → 向量化 → 存入知识库
    - 返回切分后的段落数量
    """
    # 文件大小检查（前置拦截，避免大文件打爆内存）
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            413, f"文件过大，最大支持 {MAX_UPLOAD_SIZE // 1024 // 1024}MB"
        )

    # 文件类型校验
    if not file.filename.endswith(('.txt', '.md', '.pdf', '.docx')):
        raise HTTPException(400, "仅支持 .txt、.md、.pdf 和 .docx 文件格式")

    # 读取文件内容
    raw = await file.read()

    # 文档去重：按文件内容 SHA256 检测重复上传
    content_hash = hashlib.sha256(raw).hexdigest()
    existing = await async_get_document_by_hash(content_hash)
    if existing:
        raise HTTPException(
            409,
            f"文档已存在（\"{existing['filename']}\"，{existing['chunk_count']} 个段落，"
            f"于 {existing['created_at']} 上传）。如需更新请先删除后重新上传。"
        )

    if file.filename.endswith('.pdf'):
        # PDF 处理：marker 统一提取（自动处理文字层 / 扫描件 / 表格 / 标题）
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        try:
            tmp.write(raw)
            tmp.close()
            content = extract_text(tmp.name)
            if not content.strip():
                raise HTTPException(400, "PDF 文件解析失败或内容为空")
        finally:
            os.unlink(tmp.name)

    elif file.filename.endswith('.docx'):
        # Word 处理：结构化提取（表格→Markdown表格、标题→#层级、列表→-前缀）
        from pdf_handler import extract_docx
        content = extract_docx(raw)
        if not content.strip():
            raise HTTPException(400, "Word 文件内容为空")

    else:
        # txt / md：直接读取
        content = raw.decode('utf-8')
        if not content.strip():
            raise HTTPException(400, "文件内容为空")

    # 1. 存 SQLite（记录元信息）
    doc_id = await async_save_document(file.filename, content, 0, content_hash, doc_type)

    # 2. 存 ChromaDB（切分 + Embedding + 索引）
    chunk_count = await rag_engine.async_add_document(doc_id, file.filename, content)

    # 3. 更新 SQLite 中的 chunk 数量
    await async_update_chunk_count(doc_id, chunk_count)

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        chunks=chunk_count,
        message=f"✅ 成功上传「{file.filename}」，拆分为 {chunk_count} 个段落"
    )


@app.post("/ask")
async def ask_stream(
    req: AskRequest,
    _=Depends(verify_token)
):
    """
    流式问答（SSE）— 逐 token 推送
    支持 top_k（1-20）和 temperature（0-1）
    """
    if not req.query.strip():
        raise HTTPException(400, "问题不能为空")

    async def event_generator():
        # asyncio.to_thread 避免 ChromaDB 检索阻塞事件循环
        import asyncio as _asyncio
        import json as _json

        try:
            gen = await _asyncio.to_thread(
                rag_engine.ask_stream, req.query.strip(), req.top_k, req.temperature
            )
            for event in gen:
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )


@app.get("/documents", response_model=list[DocumentItem])
async def list_documents(_=Depends(verify_token)):
    """查看已上传的文档列表"""
    docs = await async_get_documents()
    return [
        DocumentItem(
            id=d["id"],
            filename=d["filename"],
            created_at=str(d["created_at"]),
            chunk_count=d["chunk_count"],
            doc_type=d.get("doc_type", "general")
        )
        for d in docs
    ]


@app.delete("/documents")
async def clear_all_documents(_=Depends(verify_token)):
    """清空全部文档（数据库 + 向量库）"""
    docs = await async_get_documents()
    for d in docs:
        await async_delete_document(d["id"])
        await rag_engine.async_delete_document(d["id"])
    return {"message": f"已清空 {len(docs)} 个文档"}


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: int, _=Depends(verify_token)):
    """
    删除文档（先删 SQLite 记录，再删 ChromaDB 向量）
    先删数据库记录：失败则直接报错，不会出现"向量丢了但记录还在"的情况
    """
    # 先删 SQLite 记录（失败则抛异常，不会执行后续）
    if not await async_delete_document(doc_id):
        raise HTTPException(404, f"文档 #{doc_id} 不存在")

    # 再删向量库
    await rag_engine.async_delete_document(doc_id)

    return {"message": f"文档 #{doc_id} 已删除"}


@app.get("/settings")
async def get_settings(_=Depends(verify_token)):
    """返回系统配置信息（只读）"""
    from config import EMBEDDING_MODEL, LLM_MODEL, CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE, CHILD_OVERLAP, PARENT_OVERLAP, TOP_K
    docs = await async_get_documents()
    return {
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "chunk": {
            "child_size": CHILD_CHUNK_SIZE,
            "child_overlap": CHILD_OVERLAP,
            "parent_size": PARENT_CHUNK_SIZE,
            "parent_overlap": PARENT_OVERLAP,
        },
        "default_top_k": TOP_K,
        "default_temperature": 0.6,
        "document_count": len(docs),
    }


# ────────────────────── 面试 API ──────────────────────


@app.post("/interview/start")
async def interview_start(
    req: InterviewStartRequest,
    _=Depends(verify_token)
):
    """
    开始模拟面试
    请求体：{"resume_doc_id": 1, "jd_doc_id": null, "total_questions": 8}
    返回：{session_id, profile, first_question}
    """
    resume_id = req.resume_doc_id
    jd_id = req.jd_doc_id
    total_q = req.total_questions

    # 读取文档内容
    resume_text = await async_get_document_content(resume_id)
    if not resume_text:
        raise HTTPException(400, "简历文档不存在或内容为空")

    jd_text = ""
    if jd_id:
        jd_text = await async_get_document_content(jd_id) or ""

    # 创建 Session
    session_id = await async_create_interview_session(resume_id, jd_id, total_q)

    # 查同一份简历的历史题目（避免重复）
    avoid_questions = await async_get_previous_questions_by_resume(resume_id)
    if avoid_questions:
        logger.info("找到 %d 道历史题目，Agent 将避免重复提问", len(avoid_questions))

    # Agent 分析 + 出题
    try:
        result = await asyncio.to_thread(
            interview_agent.start_interview, resume_text, jd_text, total_q, avoid_questions
        )
    except Exception as e:
        logger.error("Agent 启动面试失败: %s", e)
        raise HTTPException(500, f"AI 面试官启动失败：{str(e)}")

    first_q = result["first_question"]

    # 保存第一题到 DB
    q_id = await async_add_interview_question(
        session_id, 1,
        first_q.get("question", ""),
        first_q.get("dimension"),
        first_q.get("difficulty"),
        json.dumps(first_q.get("expected_keywords", []), ensure_ascii=False),
    )

    # 更新 session 状态
    profile = result.get("profile", {})
    await async_update_interview_session(
        session_id,
        status="in_progress",
        current_question=1,
        agent_state=json.dumps(profile, ensure_ascii=False),
    )

    return {
        "session_id": session_id,
        "question_id": q_id,
        "question_number": 1,
        "total_questions": total_q,
        "profile": profile,
        "avoided_count": len(avoid_questions),
        "question": {
            "text": first_q.get("question", ""),
            "dimension": first_q.get("dimension", "技术深度"),
            "difficulty": first_q.get("difficulty", "medium"),
            "expected_keywords": first_q.get("expected_keywords", []),
            "hint": first_q.get("hint", ""),
        },
    }


@app.post("/interview/answer")
async def interview_answer(
    req: InterviewAnswerRequest,
    _=Depends(verify_token)
):
    """
    提交回答 → Agent 评分 → 下一题或完成
    请求体：{"session_id": 1, "question_id": 1, "answer": "..."}
    返回：{score, question_number, next_question, is_complete}
    """
    session_id = req.session_id
    question_id = req.question_id
    user_answer = req.answer

    session = await async_get_interview_session(session_id)
    if not session:
        raise HTTPException(404, "面试 Session 不存在")

    # 读取本轮题目信息
    questions = await async_get_interview_questions(session_id)
    current_q = None
    for q in questions:
        if q["id"] == question_id:
            current_q = q
            break
    if not current_q:
        raise HTTPException(404, "题目不存在")

    # Agent 评分 + 生成下一题
    q_num = current_q["question_number"]
    total_q = session["total_questions"]
    history_list = [
        {"q": q["question_text"], "a": q.get("user_answer", ""),
         "score": q.get("score_total"), "dimension": q.get("dimension")}
        for q in questions if q.get("user_answer")
    ]

    # 读取简历和 JD 内容（出题时参考）
    resume_text = await async_get_document_content(session["resume_doc_id"]) or ""
    jd_text = ""
    if session.get("jd_doc_id"):
        jd_text = await async_get_document_content(session["jd_doc_id"]) or ""

    # 查同一份简历的历史题目（避免重复）
    avoid_questions = await async_get_previous_questions_by_resume(session["resume_doc_id"])

    try:
        result = await asyncio.to_thread(
            interview_agent.answer_question,
            current_q["question_text"],
            current_q.get("dimension") or "技术深度",
            current_q.get("difficulty") or "medium",
            json.loads(current_q.get("expected_keywords", "[]")) if current_q.get("expected_keywords") else [],
            user_answer,
            q_num, total_q,
            history_list,
            resume_text, jd_text,
            avoid_questions,
        )
    except Exception as e:
        logger.error("Agent 评分失败: %s", e)
        raise HTTPException(500, f"评分失败：{str(e)}")

    score = result["score"]

    # 保存评分到 DB
    await async_update_interview_answer(
        question_id, user_answer,
        score.get("total_score", 5),
        json.dumps(score.get("dimensions", {}), ensure_ascii=False),
        score.get("feedback", ""),
    )

    if result["is_complete"]:
        await async_update_interview_session(session_id, status="completed")
        return {
            "score": score,
            "is_complete": True,
            "message": "面试完成！可以去查看报告了。",
        }

    # 有下一题
    next_q = result["next_question"]
    new_q_id = await async_add_interview_question(
        session_id, q_num + 1,
        next_q.get("question", ""),
        next_q.get("dimension"),
        next_q.get("difficulty"),
        json.dumps(next_q.get("expected_keywords", []), ensure_ascii=False),
    )
    await async_update_interview_session(session_id, current_question=q_num + 1)

    return {
        "score": score,
        "is_complete": False,
        "question_number": q_num + 1,
        "total_questions": total_q,
        "question_id": new_q_id,
        "question": {
            "text": next_q.get("question", ""),
            "dimension": next_q.get("dimension", "技术深度"),
            "difficulty": next_q.get("difficulty", "medium"),
            "expected_keywords": next_q.get("expected_keywords", []),
            "hint": next_q.get("hint", ""),
        },
    }


@app.get("/interview/{session_id}/status")
async def interview_status(
    session_id: int,
    _=Depends(verify_token)
):
    """查询面试进度"""
    session = await async_get_interview_session(session_id)
    if not session:
        raise HTTPException(404, "面试 Session 不存在")
    questions = await async_get_interview_questions(session_id)
    return {
        "session_id": session_id,
        "status": session["status"],
        "current_question": session["current_question"],
        "total_questions": session["total_questions"],
        "questions_count": len(questions),
        "answered_count": sum(1 for q in questions if q.get("user_answer")),
    }


@app.get("/interview/{session_id}/report")
async def interview_report(
    session_id: int,
    _=Depends(verify_token)
):
    """获取面试报告"""
    # 先从缓存查
    cached = await async_get_interview_report(session_id)
    if cached:
        return _build_report_response(session_id, cached, await async_get_interview_questions(session_id))

    # 没有缓存 → Agent 生成
    questions = await async_get_interview_questions(session_id)
    qa_history = [
        {"question_number": q["question_number"],
         "question": q["question_text"],
         "dimension": q.get("dimension"),
         "difficulty": q.get("difficulty"),
         "expected_keywords": json.loads(q.get("expected_keywords", "[]")),
         "answer": q.get("user_answer", ""),
         "score": q.get("score_total"),
         "feedback": q.get("feedback", "")}
        for q in questions if q.get("user_answer")
    ]

    if not qa_history:
        raise HTTPException(400, "还没有已回答的题目，无法生成报告")

    try:
        report = await asyncio.to_thread(
            interview_agent.generate_report, qa_history
        )
    except Exception as e:
        logger.error("Agent 生成报告失败: %s", e)
        raise HTTPException(500, f"生成报告失败：{str(e)}")

    # 缓存报告
    suggestions = report.get("learning_suggestions", [])
    await async_save_interview_report(
        session_id,
        report.get("overall_score", 0),
        json.dumps(report.get("dimension_scores", {}), ensure_ascii=False),
        report.get("strengths", ""),
        report.get("weaknesses", ""),
        "\n".join(suggestions) if isinstance(suggestions, list) else str(suggestions),
        report.get("summary", ""),
    )

    return _build_report_response(session_id, report, questions)


def _build_report_response(session_id: int, report: dict, questions: list) -> dict:
    """组装报告响应（含每题明细 + LLM 分析字段）"""
    questions_detail = [
        {
            "number": q["question_number"],
            "question": q["question_text"],
            "dimension": q.get("dimension"),
            "difficulty": q.get("difficulty"),
            "expected_keywords": json.loads(q.get("expected_keywords", "[]")) if q.get("expected_keywords") else [],
            "answer": q.get("user_answer", ""),
            "score": q.get("score_total"),
            "feedback": q.get("feedback", ""),
        }
        for q in (questions or [])
        if q.get("user_answer")
    ]
    suggestions = report.get("learning_suggestions", [])
    return {
        "session_id": session_id,
        "overall_score": report.get("overall_score", 0),
        "dimension_scores": report.get("dimension_scores", {}),
        "questions": questions_detail,
        "question_analysis": report.get("question_analysis", []),
        "keyword_analysis": report.get("keyword_analysis", {}),
        "difficulty_distribution": report.get("difficulty_distribution", {}),
        "strengths": report.get("strengths", ""),
        "weaknesses": report.get("weaknesses", ""),
        "learning_suggestions": suggestions if isinstance(suggestions, list) else str(suggestions),
        "summary": report.get("summary", ""),
    }


# ========== 直接启动（读取 config.PORT） ==========

if __name__ == "__main__":
    import uvicorn
    from config import PORT, HOST
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

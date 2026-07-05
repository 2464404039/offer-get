"""
FastAPI 主应用 — 路由定义 + 启动逻辑
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dataclasses import dataclass

import hashlib, json, asyncio, hmac, time, secrets, os, logging

from config import CHROMA_PERSIST_DIR, APP_SECRET_KEY, MAX_UPLOAD_SIZE
from schemas import (
    UploadResponse, AskRequest, AskResponse,
    SourceItem, DocumentItem,
    InterviewStartRequest, InterviewAnswerRequest,
    RegisterRequest, LoginRequest,
    SaveQuestionRequest, QuestionItem,
    StudyPlanRequest,
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
    async_create_user, async_get_user_by_username, async_get_user_by_id,
    async_toggle_user_active, async_get_all_users, async_get_user_stats,
    async_save_question, async_get_questions, async_delete_question,
    async_get_question_by_id,
)
from rag_engine import RAGEngine
from pdf_handler import extract_text
from interview_agent import InterviewAgent


logger = logging.getLogger(__name__)


# ────────────────────── 认证工具函数 ──────────────────────

def hash_password(password: str) -> str:
    """pbkdf2_hmac 密码哈希（零依赖）"""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt.hex() + ':' + dk.hex()


def verify_password(password: str, stored: str) -> bool:
    """验证密码"""
    try:
        salt_hex, dk_hex = stored.split(':')
        dk = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt_hex), 100000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


def issue_user_token(user_id: int, is_admin: bool) -> str:
    """签发用户/管理员 token"""
    prefix = "admin" if is_admin else "user"
    expiry = int(time.time()) + SESSION_TOKEN_EXPIRE
    payload = f"{prefix}:{user_id}:{expiry}"
    sig = hmac.new(APP_SECRET_KEY.encode(), payload.encode(), 'sha256').hexdigest()[:16]
    return f"{payload}:{sig}"


def _validate_user_token(token: str) -> tuple[int, bool] | None:
    """验证用户 token，返回 (user_id, is_admin) 或 None"""
    parts = token.split(":")
    if len(parts) != 4 or parts[0] not in ("user", "admin"):
        return None
    prefix, uid_str, expiry_str, sig = parts
    try:
        user_id = int(uid_str)
        expiry = int(expiry_str)
    except ValueError:
        return None
    if time.time() > expiry:
        return None
    payload = f"{prefix}:{user_id}:{expiry}"
    expected = hmac.new(APP_SECRET_KEY.encode(), payload.encode(), 'sha256').hexdigest()[:16]
    if not hmac.compare_digest(sig, expected):
        return None
    return (user_id, prefix == "admin")


# ────────────────────── 认证上下文 ──────────────────────

@dataclass
class AuthContext:
    user_id: int | None = None
    is_admin: bool = False
    is_api_key: bool = False
    is_session: bool = False

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


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(_security)) -> AuthContext:
    """
    验证 Bearer token，返回 AuthContext：
    - APP_SECRET_KEY → AuthContext(is_admin=True, is_api_key=True)
    - session token → AuthContext(is_session=True)  未登录
    - user token → AuthContext(user_id=X, is_admin=False)
    - admin token → AuthContext(user_id=X, is_admin=True)
    """
    if not APP_SECRET_KEY:
        raise HTTPException(
            500, "服务器认证未配置。请在 .env 中设置 APP_SECRET_KEY。"
        )
    token = credentials.credentials
    if token == APP_SECRET_KEY:
        return AuthContext(is_admin=True, is_api_key=True)
    if _validate_session_token(token):
        return AuthContext(is_session=True)
    result = _validate_user_token(token)
    if result:
        return AuthContext(user_id=result[0], is_admin=result[1])
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
    auth: AuthContext = Depends(verify_token)
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
        import tempfile as _tmpmod, os as _os
        tmp = _tmpmod.NamedTemporaryFile(delete=False, suffix='.pdf')
        try:
            tmp.write(raw)
            tmp.close()
            content = extract_text(tmp.name)
            if not content.strip():
                raise HTTPException(400, "PDF 文件解析失败或内容为空")
        finally:
            _os.unlink(tmp.name)

    elif file.filename.endswith('.docx'):
        from pdf_handler import extract_docx
        content = extract_docx(raw)
        if not content.strip():
            raise HTTPException(400, "Word 文件内容为空")

    else:
        content = raw.decode('utf-8')
        if not content.strip():
            raise HTTPException(400, "文件内容为空")

    # 1. 存 SQLite（记录元信息 + 用户归属）
    doc_id = await async_save_document(file.filename, content, 0, content_hash, doc_type, auth.user_id)

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
    auth: AuthContext = Depends(verify_token)
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


def _get_docs_params(auth: AuthContext) -> tuple[int | None, bool]:
    """根据认证上下文返回 (user_id, show_all)"""
    if auth.is_api_key or auth.is_admin:
        return (None, True)   # 全权限
    if auth.is_session:
        return (None, False)  # 未登录：只看公共文档
    return (auth.user_id, False)  # 登录用户：自己的 + 公共


@app.get("/documents", response_model=list[DocumentItem])
async def list_documents(auth: AuthContext = Depends(verify_token)):
    """查看已上传的文档列表"""
    uid, show_all = _get_docs_params(auth)
    docs = await async_get_documents(uid, show_all=show_all)
    return [
        DocumentItem(
            id=d["id"],
            filename=d["filename"],
            created_at=str(d["created_at"]),
            chunk_count=d["chunk_count"],
            doc_type=d.get("doc_type", "general"),
            user_id=d.get("user_id"),
        )
        for d in docs
    ]


@app.delete("/documents")
async def clear_all_documents(auth: AuthContext = Depends(verify_token)):
    """清空全部文档（数据库 + 向量库）"""
    uid, show_all = _get_docs_params(auth)
    docs = await async_get_documents(uid, show_all=show_all)
    for d in docs:
        await async_delete_document(d["id"])
        await rag_engine.async_delete_document(d["id"])
    return {"message": f"已清空 {len(docs)} 个文档"}


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: int, auth: AuthContext = Depends(verify_token)):
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
async def get_settings(auth: AuthContext = Depends(verify_token)):
    """返回系统配置信息（只读）"""
    from config import EMBEDDING_MODEL, LLM_MODEL, CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE, CHILD_OVERLAP, PARENT_OVERLAP, TOP_K
    uid, show_all = _get_docs_params(auth)
    docs = await async_get_documents(uid, show_all=show_all)
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
    auth: AuthContext = Depends(verify_token)
):
    """
    开始模拟面试
    支持从题库抽题（question_ids），剩余由 AI 补充
    隐式注入知识库上下文（自动检索面经/博客/JD）
    """
    resume_id = req.resume_doc_id
    jd_id = req.jd_doc_id
    total_q = req.total_questions
    bank_ids = req.question_ids or []

    # 读取文档内容
    resume_text = await async_get_document_content(resume_id)
    if not resume_text:
        raise HTTPException(400, "简历文档不存在或内容为空")

    jd_text = ""
    if jd_id:
        jd_text = await async_get_document_content(jd_id) or ""

    # ── 隐式注入：检索知识库中相关内容 ──
    kb_context = ""
    try:
        kb_query = f"{resume_text[:200]} {jd_text[:200]} 面试 考点 高频"
        kb_results = await asyncio.to_thread(
            rag_engine.hybrid_search, kb_query, 5
        )
        if kb_results:
            kb_context = "\n".join(r["content"][:300] for r in kb_results)
            logger.info("知识库注入: %d 条参考", len(kb_results))
    except Exception as e:
        logger.warning("知识库检索失败，跳过注入: %s", e)

    # ── 题库抽题 ──
    bank_questions = []
    if bank_ids:
        for bid in bank_ids:
            bq = await async_get_question_by_id(bid)
            if bq:
                bank_questions.append(bq)
        logger.info("从题库抽取 %d 题", len(bank_questions))

    # 创建 Session
    session_id = await async_create_interview_session(
        resume_id, jd_id, total_q, auth.user_id
    )

    # 查同一份简历的历史题目（避免重复）
    avoid_questions = await async_get_previous_questions_by_resume(resume_id)
    if avoid_questions:
        logger.info("找到 %d 道历史题目，Agent 将避免重复提问", len(avoid_questions))

    # 决定第一题来源
    if bank_questions:
        # 第一题从题库取
        bq = bank_questions[0]
        first_q = {
            "question": bq["question_text"],
            "dimension": bq.get("dimension") or "技术深度",
            "difficulty": bq.get("difficulty") or "medium",
            "expected_keywords": [],
            "hint": "",
        }
        profile = {"skills": [], "experience_years": 0, "gaps": [], "recommended_dimensions": ["技术深度"]}
        # 剩余题库题存入 session agent_state 供后续使用
        remaining_bank = bank_questions[1:]
        logger.info("第一题来自题库，剩余 %d 题待出", len(remaining_bank))
    else:
        # Agent 分析 + 出题
        try:
            result = await asyncio.to_thread(
                interview_agent.start_interview,
                resume_text, jd_text, total_q, avoid_questions,
                kb_context=kb_context,
            )
        except Exception as e:
            logger.error("Agent 启动面试失败: %s", e)
            raise HTTPException(500, f"AI 面试官启动失败：{str(e)}")
        first_q = result["first_question"]
        profile = result.get("profile", {})
        remaining_bank = []

    # 保存第一题到 DB
    q_id = await async_add_interview_question(
        session_id, 1,
        first_q.get("question", ""),
        first_q.get("dimension"),
        first_q.get("difficulty"),
        json.dumps(first_q.get("expected_keywords", []), ensure_ascii=False),
    )

    # 更新 session 状态（含剩余题库题）
    state = {
        "profile": profile if not bank_questions else profile,
        "remaining_bank": [{"id": b["id"], "text": b["question_text"],
                            "dimension": b.get("dimension"), "difficulty": b.get("difficulty")}
                           for b in remaining_bank],
    }
    await async_update_interview_session(
        session_id,
        status="in_progress",
        current_question=1,
        agent_state=json.dumps(state, ensure_ascii=False),
    )

    return {
        "session_id": session_id,
        "question_id": q_id,
        "question_number": 1,
        "total_questions": total_q,
        "profile": profile,
        "avoided_count": len(avoid_questions),
        "bank_questions_used": len(bank_questions),
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
    auth: AuthContext = Depends(verify_token)
):
    """
    提交回答 → Agent 评分 → 下一题或完成
    下一题优先从题库剩余题取，其次 AI 生成
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

    q_num = current_q["question_number"]
    total_q = session["total_questions"]

    # Agent 评分
    kw_json = json.loads(current_q.get("expected_keywords", "[]")) if current_q.get("expected_keywords") else []
    try:
        score_result = await asyncio.to_thread(
            interview_agent._exec_score,
            current_q["question_text"],
            current_q.get("dimension") or "技术深度",
            json.dumps(kw_json, ensure_ascii=False),
            user_answer,
        )
    except Exception as e:
        logger.error("Agent 评分失败: %s", e)
        raise HTTPException(500, f"评分失败：{str(e)}")

    score = score_result

    # 保存评分到 DB
    await async_update_interview_answer(
        question_id, user_answer,
        score.get("total_score", 5),
        json.dumps(score.get("dimensions", {}), ensure_ascii=False),
        score.get("feedback", ""),
    )

    if q_num >= total_q:
        await async_update_interview_session(session_id, status="completed")
        return {
            "score": score,
            "is_complete": True,
            "message": "面试完成！可以去查看报告了。",
        }

    # ── 生成下一题 ──
    history_list = [
        {"q": q["question_text"], "a": q.get("user_answer", ""),
         "score": q.get("score_total"), "dimension": q.get("dimension")}
        for q in questions if q.get("user_answer")
    ]

    # 检查是否有剩余题库题
    agent_state = json.loads(session.get("agent_state", "{}")) if session.get("agent_state") else {}
    remaining_bank = agent_state.get("remaining_bank", [])

    if remaining_bank:
        # 从题库取下一题
        bq = remaining_bank.pop(0)
        next_q = {
            "question": bq["text"],
            "dimension": bq.get("dimension") or "技术深度",
            "difficulty": bq.get("difficulty") or "medium",
            "expected_keywords": [],
            "hint": "",
        }
        # 更新 agent_state 去掉已用题
        agent_state["remaining_bank"] = remaining_bank
        await async_update_interview_session(
            session_id, current_question=q_num + 1,
            agent_state=json.dumps(agent_state, ensure_ascii=False),
        )
    else:
        # AI 生成下一题（含知识库注入）
        resume_text = await async_get_document_content(session["resume_doc_id"]) or ""
        jd_text = ""
        if session.get("jd_doc_id"):
            jd_text = await async_get_document_content(session["jd_doc_id"]) or ""

        kb_context = ""
        try:
            kb_query = f"{resume_text[:200]} {jd_text[:200]} 面试 考点"
            kb_results = await asyncio.to_thread(rag_engine.hybrid_search, kb_query, 5)
            if kb_results:
                kb_context = "\n".join(r["content"][:300] for r in kb_results)
        except Exception as e:
            logger.warning("知识库检索失败: %s", e)

        avoid_questions = await async_get_previous_questions_by_resume(session["resume_doc_id"])
        history_json = json.dumps(history_list, ensure_ascii=False)

        try:
            next_q_raw = await asyncio.to_thread(
                interview_agent._exec_question,
                current_q.get("dimension") or "技术深度",
                current_q.get("difficulty") or "medium",
                history_json,
                resume_text, jd_text,
                avoid_questions,
                kb_context=kb_context,
            )
        except Exception as e:
            logger.error("Agent 出题失败: %s", e)
            raise HTTPException(500, f"出题失败：{str(e)}")
        next_q = next_q_raw

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
    auth: AuthContext = Depends(verify_token)
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
    auth: AuthContext = Depends(verify_token)
):
    """获取面试报告"""
    # 先从缓存查
    cached = await async_get_interview_report(session_id)
    questions = await async_get_interview_questions(session_id)
    if cached:
        return _build_report_response(session_id, cached, questions)

    # 没有缓存 → Agent 生成
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

    # ── 反馈闭环：找最弱维度 → 检索知识库推荐 ──
    prep_recommendations = []
    dim_scores = report.get("dimension_scores", {})
    if dim_scores:
        weakest = min(dim_scores, key=dim_scores.get)
        try:
            prep_docs = await asyncio.to_thread(
                rag_engine.hybrid_search, f"{weakest} 学习 面试 提升", 3
            )
            prep_recommendations = [
                {"filename": d["filename"], "snippet": d["content"][:200]}
                for d in prep_docs
            ]
        except Exception as e:
            logger.warning("备战推荐检索失败: %s", e)

    return _build_report_response(session_id, report, questions, prep_recommendations)


def _build_report_response(session_id: int, report: dict, questions: list,
                          prep_recommendations: list | None = None) -> dict:
    """组装报告响应（含每题明细 + LLM 分析字段 + 备战推荐）"""
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
    result = {
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
    if prep_recommendations:
        result["prep_recommendations"] = prep_recommendations
    return result


# ────────────────────── 认证 API ──────────────────────


@app.post("/auth/register")
async def auth_register(req: RegisterRequest):
    """用户注册"""
    if not APP_SECRET_KEY:
        raise HTTPException(500, "服务器认证未配置")
    existing = await async_get_user_by_username(req.username)
    if existing:
        raise HTTPException(409, "用户名已存在")
    pwd_hash = hash_password(req.password)
    user_id = await async_create_user(req.username, pwd_hash)
    token = issue_user_token(user_id, is_admin=False)
    return {"id": user_id, "username": req.username, "token": token}


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    """用户登录"""
    if not APP_SECRET_KEY:
        raise HTTPException(500, "服务器认证未配置")
    user = await async_get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")
    if not user["is_active"]:
        raise HTTPException(403, "账号已被禁用")
    token = issue_user_token(user["id"], bool(user["is_admin"]))
    return {
        "token": token,
        "user": {"id": user["id"], "username": user["username"], "is_admin": bool(user["is_admin"])},
    }


@app.get("/auth/me")
async def auth_me(auth: AuthContext = Depends(verify_token)):
    """当前用户信息"""
    if auth.is_session:
        return {"authenticated": False, "message": "未登录（session 模式）"}
    if auth.is_api_key:
        return {"authenticated": True, "username": "API Key", "is_admin": True}
    user = await async_get_user_by_id(auth.user_id)
    if not user:
        raise HTTPException(404, "用户不存在")
    return {"id": user["id"], "username": user["username"], "is_admin": bool(user["is_admin"])}


# ────────────────────── 备战 API ──────────────────────


@app.post("/prep/analyze")
async def prep_analyze(auth: AuthContext = Depends(verify_token)):
    """一键全局分析：基于知识库分析面试趋势、高频考点"""
    async def event_generator():
        try:
            gen = await asyncio.to_thread(
                rag_engine.ask_stream,
                "请全面分析知识库中的所有面经、JD和技术博客，给出：1.目标公司/岗位的高频考点Top5 2.面试流程特点 3.常见翻车点和应对策略 4.与简历的差距分析",
                15, 0.6, "prep"
            )
            for event in gen:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.post("/prep/study-plan")
async def prep_study_plan(
    req: StudyPlanRequest,
    auth: AuthContext = Depends(verify_token)
):
    """生成备战建议"""
    resume_context = ""
    if req.resume_doc_id:
        resume_text = await async_get_document_content(req.resume_doc_id)
        if resume_text:
            resume_context = f"\n候选人简历背景：{resume_text[:500]}"

    query = f"根据知识库中的面经和JD，给出针对性的面试备战建议。包括：需要重点准备的知识领域、推荐的复习资料（从知识库中标出）、常见题型和应对策略、以及如何弥补简历中的短板。不要限定具体天数，根据实际情况给出合理建议。{resume_context}"

    async def event_generator():
        try:
            gen = await asyncio.to_thread(
                rag_engine.ask_stream, query, 15, 0.6, "prep"
            )
            for event in gen:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ────────────────────── 题库 API ──────────────────────


@app.post("/questions")
async def save_to_bank(
    req: SaveQuestionRequest,
    auth: AuthContext = Depends(verify_token)
):
    """收藏题目到题库"""
    qid = await async_save_question(
        auth.user_id, req.question_text,
        req.dimension, req.difficulty, "manual"
    )
    return {"id": qid, "message": "已收藏到题库"}


@app.get("/questions", response_model=list[QuestionItem])
async def list_questions(auth: AuthContext = Depends(verify_token)):
    """题库列表"""
    questions = await async_get_questions(auth.user_id)
    return [
        QuestionItem(
            id=q["id"],
            question_text=q["question_text"],
            dimension=q.get("dimension"),
            difficulty=q.get("difficulty", "medium"),
            source=q.get("source", "manual"),
            created_at=str(q["created_at"]),
        )
        for q in questions
    ]


@app.delete("/questions/{q_id}")
async def remove_question(q_id: int, auth: AuthContext = Depends(verify_token)):
    """删除题库题目"""
    if not await async_delete_question(q_id):
        raise HTTPException(404, "题目不存在")
    return {"message": f"题目 #{q_id} 已删除"}


# ────────────────────── 管理 API（仅 admin） ──────────────────────


def _require_admin(auth: AuthContext):
    if not auth.is_admin:
        raise HTTPException(403, "需要管理员权限")


@app.get("/admin/stats")
async def admin_stats(auth: AuthContext = Depends(verify_token)):
    """平台统计"""
    _require_admin(auth)
    return await async_get_user_stats()


@app.get("/admin/users")
async def admin_users(auth: AuthContext = Depends(verify_token)):
    """用户列表"""
    _require_admin(auth)
    return await async_get_all_users()


@app.put("/admin/users/{user_id}/toggle-active")
async def admin_toggle_user(user_id: int, auth: AuthContext = Depends(verify_token)):
    """启用/禁用用户"""
    _require_admin(auth)
    if not await async_toggle_user_active(user_id):
        raise HTTPException(404, "用户不存在")
    user = await async_get_user_by_id(user_id)
    status = "启用" if user["is_active"] else "禁用"
    return {"message": f"用户 {user['username']} 已{status}"}


@app.get("/admin/interviews")
async def admin_interviews(auth: AuthContext = Depends(verify_token)):
    """全部面试记录（含用户信息）"""
    _require_admin(auth)
    # 复用现有查询（无 user_id 过滤 = 全量）
    from database import get_connection as _get_conn
    conn = await asyncio.to_thread(_get_conn)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.id, s.resume_doc_id, s.status, s.total_questions,
               s.current_question, s.created_at,
               u.username, COALESCE(r.overall_score, 0) as score
        FROM interview_sessions s
        LEFT JOIN users u ON s.user_id = u.id
        LEFT JOIN interview_reports r ON s.id = r.session_id
        ORDER BY s.created_at DESC
        LIMIT 100
    ''')
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    return rows


# ────────────────────── 管理端页面 ──────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """管理控制台页面"""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "admin.html")
    if not os.path.exists(template_path):
        raise HTTPException(404, "管理页面不存在")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


# ========== 直接启动（读取 config.PORT） ==========

if __name__ == "__main__":
    import uvicorn
    from config import PORT, HOST
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

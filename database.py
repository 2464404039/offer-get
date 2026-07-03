"""
数据库操作层 — SQLite（零依赖，开箱即用）

所有函数原生同步，通过 @_make_async 装饰器导出异步版本。
生产部署可替换为 MySQL / PostgreSQL，改动仅限于本文件。
"""

import asyncio
import functools
import sqlite3
import logging
import threading
from config import SQLITE_PATH

logger = logging.getLogger(__name__)


# ========== 装饰器：同步函数 → 异步函数（一刀替代 ~90 行 async wrapper） ==========

def _make_async(fn):
    """将同步 DB 函数包装为异步函数（通过 asyncio.to_thread 规避 GIL 阻塞）"""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)
    return wrapper


# ========== 连接管理 ==========
# 长连接 + WAL 模式：读不阻塞写，写不阻塞读
# ✓ threading.Lock 保护 _conn 初始化：多线程 / 多 worker 场景安全
# × 仍为单连接，生产部署建议替换为连接池（如 SQLAlchemy 或 aiosqlite）

_conn = None
_conn_lock = threading.Lock()


def get_connection():
    global _conn
    if _conn is None:
        with _conn_lock:
            if _conn is None:
                _conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
                _conn.row_factory = sqlite3.Row
                _conn.execute("PRAGMA journal_mode=WAL")
                _conn.execute("PRAGMA busy_timeout=5000")
                _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


# ========== 建表 ==========

def init_db():
    """应用启动时创建表"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT,
            chunk_count INTEGER DEFAULT 0,
            doc_type TEXT DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 为旧数据库补 content_hash 列（无痛升级）
    try:
        cursor.execute("SELECT content_hash FROM documents LIMIT 0")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")

    # 为旧数据库补 doc_type 列（无痛升级）
    try:
        cursor.execute("SELECT doc_type FROM documents LIMIT 0")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE documents ADD COLUMN doc_type TEXT DEFAULT 'general'")

    # — 面试 Session 表 —
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interview_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_doc_id INTEGER NOT NULL,
            jd_doc_id INTEGER,
            status TEXT DEFAULT 'pending',
            total_questions INTEGER DEFAULT 8,
            current_question INTEGER DEFAULT 0,
            agent_state TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (resume_doc_id) REFERENCES documents(id)
        )
    ''')
    # — 面试题目&回答表 —
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interview_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question_number INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            dimension TEXT,
            difficulty TEXT DEFAULT 'medium',
            expected_keywords TEXT DEFAULT '[]',
            user_answer TEXT,
            score_total REAL,
            score_detail TEXT,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES interview_sessions(id)
        )
    ''')

    # 为旧数据库补 expected_keywords 列（无痛升级）
    try:
        cursor.execute("SELECT expected_keywords FROM interview_questions LIMIT 0")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE interview_questions ADD COLUMN expected_keywords TEXT DEFAULT '[]'")
    # — 面试报告表 —
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interview_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER UNIQUE NOT NULL,
            overall_score REAL,
            dimension_scores TEXT,
            strengths TEXT,
            weaknesses TEXT,
            learning_suggestions TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES interview_sessions(id)
        )
    ''')

    conn.commit()
    cursor.close()
    logger.info("数据库表初始化完成（含面试表）")


# ========== 文档操作 ==========

def save_document(filename: str, content: str, chunk_count: int, content_hash: str = "", doc_type: str = "general") -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO documents (filename, content, chunk_count, content_hash, doc_type) VALUES (?, ?, ?, ?, ?)',
        (filename, content, chunk_count, content_hash, doc_type)
    )
    conn.commit()
    doc_id = cursor.lastrowid
    cursor.close()
    return doc_id


def get_document_by_hash(content_hash: str) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, filename, chunk_count, created_at FROM documents WHERE content_hash = ?',
        (content_hash,)
    )
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


def get_documents() -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, filename, chunk_count, doc_type, created_at FROM documents ORDER BY created_at DESC'
    )
    rows = [dict(row) for row in cursor.fetchall()]
    cursor.close()
    return rows


def delete_document(doc_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
    affected = cursor.rowcount
    conn.commit()
    cursor.close()
    return affected > 0


def update_chunk_count(doc_id: int, chunk_count: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE documents SET chunk_count = ? WHERE id = ?', (chunk_count, doc_id)
    )
    conn.commit()
    cursor.close()


# ========== 面试操作 ==========

def get_document_content(doc_id: int) -> str | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT content FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()
    cursor.close()
    return row['content'] if row else None


# ── Interview Session ──

def create_interview_session(resume_doc_id: int, jd_doc_id: int | None,
                             total_questions: int = 8) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO interview_sessions (resume_doc_id, jd_doc_id, total_questions) VALUES (?, ?, ?)',
        (resume_doc_id, jd_doc_id, total_questions)
    )
    conn.commit()
    sid = cursor.lastrowid
    cursor.close()
    return sid


def get_interview_session(session_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM interview_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


def update_interview_session(session_id: int, **kwargs):
    if not kwargs:
        return
    sets = ', '.join(f'{k} = ?' for k in kwargs)
    vals = list(kwargs.values()) + [session_id]
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f'UPDATE interview_sessions SET {sets} WHERE id = ?', vals)
    conn.commit()
    cursor.close()


# ── Interview Questions ──

def add_interview_question(session_id: int, question_number: int,
                           question_text: str, dimension: str | None = None,
                           difficulty: str = 'medium',
                           expected_keywords: str = '[]') -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO interview_questions (session_id, question_number, question_text, dimension, difficulty, expected_keywords) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (session_id, question_number, question_text, dimension, difficulty, expected_keywords)
    )
    conn.commit()
    qid = cursor.lastrowid
    cursor.close()
    return qid


def update_interview_answer(q_id: int, user_answer: str,
                            score_total: float, score_detail: str, feedback: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE interview_questions SET user_answer = ?, score_total = ?, score_detail = ?, feedback = ? '
        'WHERE id = ?',
        (user_answer, score_total, score_detail, feedback, q_id)
    )
    conn.commit()
    cursor.close()


def get_interview_questions(session_id: int) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM interview_questions WHERE session_id = ? ORDER BY question_number',
        (session_id,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    return rows


def get_previous_questions_by_resume(resume_doc_id: int) -> list[str]:
    """查同一份简历在历史面试中问过的题目（避免重复出题）"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT iq.question_text FROM interview_questions iq
        JOIN interview_sessions s ON iq.session_id = s.id
        WHERE s.resume_doc_id = ? AND iq.user_answer IS NOT NULL
        ORDER BY iq.id DESC
    ''', (resume_doc_id,))
    rows = [row['question_text'] for row in cursor.fetchall()]
    cursor.close()
    return rows


# ── Interview Report ──

def save_interview_report(session_id: int, overall_score: float, dimension_scores: str,
                          strengths: str, weaknesses: str,
                          learning_suggestions: str, summary: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT OR REPLACE INTO interview_reports '
        '(session_id, overall_score, dimension_scores, strengths, weaknesses, learning_suggestions, summary) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        (session_id, overall_score, dimension_scores, strengths, weaknesses, learning_suggestions, summary)
    )
    conn.commit()
    cursor.close()


def get_interview_report(session_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM interview_reports WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


# ========== 导出异步版本（自动生成，一行一个） ==========

async_init_db = _make_async(init_db)
async_save_document = _make_async(save_document)
async_get_document_by_hash = _make_async(get_document_by_hash)
async_get_documents = _make_async(get_documents)
async_delete_document = _make_async(delete_document)
async_update_chunk_count = _make_async(update_chunk_count)
async_get_document_content = _make_async(get_document_content)
async_create_interview_session = _make_async(create_interview_session)
async_get_interview_session = _make_async(get_interview_session)
async_update_interview_session = _make_async(update_interview_session)
async_add_interview_question = _make_async(add_interview_question)
async_update_interview_answer = _make_async(update_interview_answer)
async_get_interview_questions = _make_async(get_interview_questions)
async_get_previous_questions_by_resume = _make_async(get_previous_questions_by_resume)
async_save_interview_report = _make_async(save_interview_report)
async_get_interview_report = _make_async(get_interview_report)

"""
数据库操作层 — SQLite（零依赖，开箱即用）

提供同步函数（内部使用）和异步函数（FastAPI 路由使用）
生产部署可替换为 MySQL / PostgreSQL，改动仅限于本文件
"""

import asyncio
import sqlite3
import logging
from config import SQLITE_PATH

logger = logging.getLogger(__name__)


# ========== 连接管理 ==========
# 长连接 + WAL 模式：读不阻塞写，写不阻塞读
# 不上锁：Python GIL + SQLite 单写者足以应对 demo 并发

_conn = None


def get_connection():
    global _conn
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 为旧数据库补 content_hash 列（无痛升级）
    try:
        cursor.execute("SELECT content_hash FROM documents LIMIT 0")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")

    conn.commit()
    cursor.close()
    logger.info("数据库表初始化完成")


# ========== 文档操作 ==========

def save_document(filename: str, content: str, chunk_count: int, content_hash: str = "") -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO documents (filename, content, chunk_count, content_hash) VALUES (?, ?, ?, ?)',
        (filename, content, chunk_count, content_hash)
    )
    conn.commit()
    doc_id = cursor.lastrowid
    cursor.close()
    return doc_id


def get_document_by_hash(content_hash: str) -> dict | None:
    """按内容哈希查重，返回已存在的文档记录或 None"""
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
        'SELECT id, filename, chunk_count, created_at FROM documents ORDER BY created_at DESC'
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


# ========== 异步包装（FastAPI 路由使用） ==========

async def async_init_db():
    await asyncio.to_thread(init_db)

async def async_save_document(filename: str, content: str, chunk_count: int, content_hash: str = "") -> int:
    return await asyncio.to_thread(save_document, filename, content, chunk_count, content_hash)

async def async_get_document_by_hash(content_hash: str) -> dict | None:
    return await asyncio.to_thread(get_document_by_hash, content_hash)

async def async_get_documents() -> list:
    return await asyncio.to_thread(get_documents)

async def async_delete_document(doc_id: int) -> bool:
    return await asyncio.to_thread(delete_document, doc_id)

async def async_update_chunk_count(doc_id: int, chunk_count: int):
    await asyncio.to_thread(update_chunk_count, doc_id, chunk_count)

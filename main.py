"""
FastAPI 主应用 — 路由定义 + 启动逻辑
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import hashlib, json
import logging

from config import CHROMA_PERSIST_DIR, APP_SECRET_KEY, MAX_UPLOAD_SIZE
from schemas import (
    UploadResponse, AskRequest, AskResponse,
    SourceItem, DocumentItem
)
from database import (
    async_init_db, async_save_document, async_get_documents,
    async_delete_document, async_update_chunk_count,
    async_get_document_by_hash
)
from rag_engine import RAGEngine
from pdf_handler import extract_text


logger = logging.getLogger(__name__)

# ────────────────────── 全局变量 ──────────────────────
# 注意：RAGEngine 内部有 sentence-transformers 模型（~80MB）
# 所以在应用启动时只加载一次，所有请求共享
rag_engine: RAGEngine = None


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

    yield

    logger.info("正在关闭 RAG 引擎...")


# ────────────────────── 创建应用 ──────────────────────

app = FastAPI(
    title="智能知识库问答 API",
    description="上传文档，基于 RAG（检索增强生成）技术进行智能问答",
    version="1.0.0",
    lifespan=lifespan
)

# ────────────────────── CORS（跨域白名单） ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8765", "http://127.0.0.1:8765"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

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
# 如果配置了 APP_SECRET_KEY，所有 API 路由需要 Authorization: Bearer ***
# 未配置时（本地开发），不开启鉴权

_security = HTTPBearer(auto_error=False)  # 不自动抛 403，我们手动处理

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(_security)):
    if not APP_SECRET_KEY:
        return  # 本地开发模式，不鉴权
    if credentials is None or credentials.credentials != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="无效或缺失 API Token")


# ────────────────────── API 路由 ──────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """智能知识库问答 — Web 界面"""
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📚 智能知识库问答</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, 'Segoe UI', sans-serif; height: 100vh; display: flex; background: #f0f2f5; }

    /* ========== 左侧栏 ========== */
    .sidebar { width: 300px; background: #1a1a2e; color: #fff; display: flex; flex-direction: column; padding: 20px; }
    .sidebar h1 { font-size: 20px; margin-bottom: 4px; }
    .sidebar .sub { font-size: 12px; color: #8892b0; margin-bottom: 20px; }

    .upload-area { border: 2px dashed #3a3a5e; border-radius: 10px; padding: 20px; text-align: center; cursor: pointer; transition: .2s; margin-bottom: 16px; }
    .upload-area:hover { border-color: #64ffda; background: rgba(100,255,218,.05); }
    .upload-area .icon { font-size: 28px; }
    .upload-area .text { font-size: 13px; color: #8892b0; margin-top: 6px; }
    .upload-area .hint { font-size: 11px; color: #5a5a7a; margin-top: 4px; }
    .upload-area.uploading { opacity: .6; pointer-events: none; }

    .doc-list { flex: 1; overflow-y: auto; }
    .doc-list .title { font-size: 12px; color: #8892b0; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
    .doc-item { display: flex; align-items: center; justify-content: space-between; padding: 8px 10px; background: #16213e; border-radius: 6px; margin-bottom: 6px; font-size: 13px; }
    .doc-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .doc-item .badge { background: #0f3460; color: #64ffda; font-size: 10px; padding: 1px 6px; border-radius: 8px; margin-left: 6px; }
    .doc-item .del { cursor: pointer; color: #e74c3c; font-size: 16px; opacity: .6; transition: .2s; background: none; border: none; }
    .doc-item .del:hover { opacity: 1; }

    /* ========== 右侧主区域 ========== */
    .main { flex: 1; display: flex; flex-direction: column; }
    .chat-header { padding: 16px 24px; background: #fff; border-bottom: 1px solid #e8e8e8; font-size: 14px; color: #666; }
    .chat-header strong { color: #1a1a2e; }

    .messages { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px; }
    .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 14px; word-wrap: break-word; }
    .msg.user { background: #1a1a2e; color: #fff; align-self: flex-end; border-bottom-right-radius: 4px; }
    .msg.bot { background: #fff; color: #333; align-self: flex-start; border-bottom-left-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
    .msg.bot pre { background: #f5f5f5; padding: 8px; border-radius: 6px; overflow-x: auto; font-size: 13px; margin: 8px 0; }
    .msg.bot code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 13px; }
    .msg.bot table { border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 13px; }
    .msg.bot th, .msg.bot td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
    .msg.bot th { background: #f5f5f5; }

    .sources { margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; font-size: 12px; }
    .sources .label { color: #999; }
    .source-item { color: #1a73e8; margin: 2px 0; }

    .typing { display: flex; gap: 4px; padding: 12px 16px; }
    .typing span { width: 6px; height: 6px; background: #ccc; border-radius: 50%; animation: bounce 1.4s infinite; }
    .typing span:nth-child(2) { animation-delay: .2s; }
    .typing span:nth-child(3) { animation-delay: .4s; }
    @keyframes bounce { 0%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }

    .empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #bbb; }
    .empty-state .icon { font-size: 48px; margin-bottom: 12px; }
    .empty-state .text { font-size: 14px; }

    .input-area { padding: 16px 24px; background: #fff; border-top: 1px solid #e8e8e8; display: flex; gap: 10px; }
    .input-area input { flex: 1; padding: 10px 16px; border: 1px solid #ddd; border-radius: 24px; font-size: 14px; outline: none; transition: .2s; }
    .input-area input:focus { border-color: #1a1a2e; }
    .input-area button { width: 40px; height: 40px; border: none; border-radius: 50%; background: #1a1a2e; color: #fff; font-size: 18px; cursor: pointer; transition: .2s; display: flex; align-items: center; justify-content: center; }
    .input-area button:hover { background: #16213e; }
    .input-area button:disabled { background: #ccc; cursor: not-allowed; }

    .toast { position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%); background: #333; color: #fff; padding: 10px 24px; border-radius: 8px; font-size: 13px; opacity: 0; transition: .3s; pointer-events: none; z-index: 999; }
    .toast.show { opacity: 1; }

    /* 滚动条 */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }

    /* ========== 设置面板 ========== */
    .gear-btn { float: right; cursor: pointer; font-size: 18px; opacity: .6; transition: .2s; user-select: none; }
    .gear-btn:hover { opacity: 1; }
    .settings-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.4); z-index: 100; }
    .settings-overlay.show { display: block; }
    .settings-panel { position: fixed; top: 0; right: -360px; width: 340px; height: 100%; background: #fff;
      z-index: 101; transition: right .3s; box-shadow: -2px 0 12px rgba(0,0,0,.1); display: flex; flex-direction: column; }
    .settings-panel.show { right: 0; }
    .settings-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; border-bottom: 1px solid #eee; font-size: 16px; }
    .settings-close { cursor: pointer; font-size: 18px; opacity: .5; transition: .2s; }
    .settings-close:hover { opacity: 1; }
    .settings-body { padding: 20px; flex: 1; overflow-y: auto; }
    .setting-row { margin-bottom: 20px; }
    .setting-row label { display: block; font-size: 13px; color: #555; margin-bottom: 6px; }
    .setting-row input[type=range] { width: 100%; accent-color: #1a1a2e; }
    .danger-btn { width: 100%; padding: 10px; border: 1px solid #e74c3c; background: #fff; color: #e74c3c; border-radius: 8px;
      cursor: pointer; font-size: 14px; transition: .2s; margin-bottom: 16px; }
    .danger-btn:hover { background: #e74c3c; color: #fff; }
    .sys-info { font-size: 12px; color: #999; line-height: 1.8; padding-top: 12px; border-top: 1px solid #eee; }
</style>
</head>
<body>

<div class="sidebar">
  <h1>📚 知识库问答</h1>
  <div class="sub">RAG · 智能检索 · LLM 生成</div>

  <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
    <div class="icon">📄</div>
    <div class="text">点击上传文档</div>
    <div class="hint">支持 .txt .md .pdf .docx</div>
    <input type="file" id="fileInput" accept=".txt,.md,.pdf,.docx" style="display:none" onchange="uploadFile(this)">
  </div>

  <div class="doc-list">
    <div class="title">已上传文档</div>
    <div id="docList"></div>
  </div>
</div>

<div class="main">
  <div class="chat-header">💬 <strong>智能问答</strong> — 基于已上传文档的内容 <span class="gear-btn" onclick="toggleSettings()" title="设置">⚙</span></div>

  <div class="messages" id="messages">
    <div class="empty-state" id="emptyState">
      <div class="icon">💡</div>
      <div class="text">上传文档后，在这里提问</div>
    </div>
  </div>

  <div class="input-area">
    <input type="text" id="queryInput" placeholder="输入你的问题..." onkeydown="if(event.key==='Enter') sendQuery()">
    <button id="sendBtn" onclick="sendQuery()">➤</button>
  </div>
</div>

<div class="toast" id="toast"></div>

<!-- ⚙ 设置面板 -->
<div class="settings-overlay" id="settingsOverlay" onclick="toggleSettings()"></div>
<div class="settings-panel" id="settingsPanel">
  <div class="settings-header">
    <strong>⚙ 设置</strong>
    <span class="settings-close" onclick="toggleSettings()">✕</span>
  </div>

  <div class="settings-body">
    <div class="setting-row">
      <label>检索数量 (Top-K)：<span id="topKVal">5</span></label>
      <input type="range" id="topKSlider" min="1" max="20" value="5" oninput="updateTopK(this.value)">
    </div>

    <div class="setting-row">
      <label>回答温度 (Temperature)：<span id="tempVal">0.6</span></label>
      <input type="range" id="tempSlider" min="0" max="1" step="0.1" value="0.6" oninput="updateTemp(this.value)">
    </div>

    <button class="danger-btn" onclick="clearAllDocs()">🗑 清空知识库</button>

    <div class="sys-info" id="sysInfo">加载中...</div>
  </div>
</div>

</body>

<script>
const API = '';
let uploading = false;
let pendingQuery = false;
let lastQuery = '';
let topK = 5;
let temp = 0.6;

// getHeaders 始终可用（鉴权开启时 API_TOKEN 由后端注入）
function getHeaders() {
  return typeof API_TOKEN !== 'undefined' ? { 'Authorization': 'Bearer ' + API_TOKEN } : {};
}

// ========== Toast ==========
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

// ========== 上传文档 ==========
async function uploadFile(input) {
  const file = input.files[0];
  if (!file || uploading) return;
  uploading = true;
  if (!file.name.endsWith('.txt') && !file.name.endsWith('.md') && !file.name.endsWith('.pdf') && !file.name.endsWith('.docx')) {
    showToast('仅支持 .txt、.md、.pdf 和 .docx 文件'); return;
  }
  const area = document.getElementById('uploadArea');
  area.classList.add('uploading');
  area.querySelector('.text').textContent = '上传中...';

  const form = new FormData();
  form.append('file', file);

  try {
    const resp = await fetch(API + '/upload', { method: 'POST', body: form, headers: getHeaders() });
    const data = await resp.json();
    if (resp.ok) {
      showToast('✅ ' + data.message);
      loadDocuments();
    } else {
      showToast('❌ ' + (data.detail || '上传失败'));
    }
  } catch(e) {
    showToast('❌ 网络错误');
  }
  area.classList.remove('uploading');
  area.querySelector('.text').textContent = '点击上传文档';
  input.value = '';
}

// ========== 文档列表 ==========
async function loadDocuments() {
  try {
    const resp = await fetch(API + '/documents', { headers: getHeaders() });
    const docs = await resp.json();
    const el = document.getElementById('docList');
    if (!docs.length) { el.innerHTML = '<div style="font-size:12px;color:#5a5a7a;text-align:center;padding:20px 0">暂无文档</div>'; return; }
    el.innerHTML = docs.map(d =>
      `<div class="doc-item">
        <span class="name">📄 ${d.filename}</span>
        <span class="badge">${d.chunk_count}段</span>
        <button class="del" onclick="deleteDoc(${d.id})">✕</button>
      </div>`
    ).join('');
  } catch(e) { /* ignore */ }
}

// ========== 删除文档 ==========
async function deleteDoc(id) {
  try {
    const resp = await fetch(API + '/documents/' + id, { method: 'DELETE', headers: getHeaders() });
    if (resp.ok) { showToast('已删除'); loadDocuments(); }
    else showToast('❌ 删除失败');
  } catch(e) { showToast('❌ 网络错误'); }
}

// ========== 发送提问（流式 SSE） ==========
async function sendQuery() {
  const input = document.getElementById('queryInput');
  const query = input.value.trim();
  if (!query || uploading || pendingQuery) return;

  // 相同问题防重复
  if (query === lastQuery) {
    showToast('这个问题刚刚问过');
    return;
  }

  input.value = '';
  pendingQuery = true;
  document.getElementById('sendBtn').disabled = true;
  hideEmptyState();

  // 显示用户消息
  addMessage(query, 'user');

  // 创建空消息容器，流式填充
  const stream = createStreamingMessage();

  try {
    const resp = await fetch(API + '/ask', {
      method: 'POST',
      headers: { ...getHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK, temperature: temp })
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      stream.append('❌ ' + (err.detail || '请求失败'));
      stream.finish();
      return;
    }

    // 读取 SSE 流
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === 'token') {
            stream.append(event.content);
          } else if (event.type === 'sources') {
            stream.finish(event.sources);
          } else if (event.type === 'error') {
            stream.append('\\n❌ ' + event.message);
            stream.finish();
          } else if (event.type === 'done') {
            stream.finish();
          }
        } catch(e) { /* 忽略解析错误 */ }
      }
    }
  } catch(e) {
    stream.append('\\n❌ 网络错误：' + (e.message || '请检查服务是否启动'));
    stream.finish();
  }
  pendingQuery = false;
  lastQuery = query;
  document.getElementById('sendBtn').disabled = false;
}

// ========== 消息渲染 ==========
function addMessage(text, role) {
  const el = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ========== 流式消息（SSE 渐进渲染） ==========
function createStreamingMessage() {
  const el = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg bot';
  el.appendChild(div);

  let content = '';

  return {
    div: div,
    append: function(text) {
      content += text;
      let safe = escapeHtml(content)
        .replace(/\\n/g, '<br>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
      div.innerHTML = safe;
      el.scrollTop = el.scrollHeight;
    },
    finish: function(sources) {
      if (sources && sources.length) {
        let sHtml = '<div class="sources"><div class="label">📎 参考来源：</div>';
        const seen = new Set();
        sources.forEach(s => {
          if (!seen.has(s.document)) {
            seen.add(s.document);
            sHtml += '<div class="source-item">' + s.document + '</div>';
          }
        });
        sHtml += '</div>';
        div.innerHTML += sHtml;
      }
      el.scrollTop = el.scrollHeight;
    }
  };
}

let typingIndex = 0;

function showTyping() {
  const el = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg bot typing';
  div.id = 'typing-' + (++typingIndex);
  div.innerHTML = '<span></span><span></span><span></span>';
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return typingIndex;
}

function removeTyping(id) {
  const el = document.getElementById('typing-' + id);
  if (el) el.remove();
}

function hideEmptyState() {
  const es = document.getElementById('emptyState');
  if (es) es.style.display = 'none';
}

// ========== 初始化 ==========
loadDocuments();
loadSettings();

// ========== 设置面板 ==========
function toggleSettings() {
  const ov = document.getElementById('settingsOverlay');
  const pn = document.getElementById('settingsPanel');
  const show = !pn.classList.contains('show');
  ov.classList.toggle('show', show);
  pn.classList.toggle('show', show);
}

function updateTopK(v) {
  topK = parseInt(v);
  document.getElementById('topKVal').textContent = v;
}

function updateTemp(v) {
  temp = parseFloat(v);
  document.getElementById('tempVal').textContent = v;
}

async function clearAllDocs() {
  if (!confirm('确定清空全部文档和知识库？此操作不可恢复。')) return;
  try {
    const resp = await fetch(API + '/documents', { method: 'DELETE', headers: getHeaders() });
    if (resp.ok) { showToast('✅ 知识库已清空'); loadDocuments(); }
    else showToast('❌ 清空失败');
  } catch(e) { showToast('❌ 网络错误'); }
}

async function loadSettings() {
  try {
    const resp = await fetch(API + '/settings', { headers: getHeaders() });
    const data = await resp.json();
    document.getElementById('sysInfo').innerHTML =
      '<b>Embedding:</b> ' + data.embedding_model.split('/').pop() + '<br>' +
      '<b>LLM:</b> ' + data.llm_model + '<br>' +
      '<b>切块:</b> Child ' + data.chunk.child_size + '字 / Parent ' + data.chunk.parent_size + '字<br>' +
      '<b>文档数:</b> ' + data.document_count;
  } catch(e) { /* ignore */ }
}
</script>
</html>"""

    # 注入 API Token（如果配置了认证）
    token_js = ""
    if APP_SECRET_KEY:
        # json.dumps 安全转义特殊字符（如单引号、反斜杠），避免 JS 注入
        safe_key = json.dumps(APP_SECRET_KEY)
        token_js = f"const API_TOKEN = {safe_key};"
        html = html.replace("const API = '';", f"const API = '';\n{token_js}")

    return html


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
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
    doc_id = await async_save_document(file.filename, content, 0, content_hash)

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
            chunk_count=d["chunk_count"]
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


# ========== 直接启动（读取 config.PORT） ==========

if __name__ == "__main__":
    import uvicorn
    from config import PORT, HOST
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

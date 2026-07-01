# 📚 智能知识库问答 API

基于 **RAG（检索增强生成）** 技术的文档智能问答系统。上传 PDF / Word / Markdown / 纯文本 文档，自然语言提问，流式返回基于文档内容的回答。

---

## 核心设计

### Parent-Child 双层切块

```
文档 → Parent Chunk (2000字, 语义完整)
     → Child Chunk × N (300字, 精确检索)

提问 → Child 级检索（精度高）→ 收集 Parent ID → 返回完整 Parent 内容 → LLM 生成
```

解决了传统单层切块的「精度 vs 完整性」矛盾：小块检索准，大块给 LLM 上下文完整。

### 流式输出（SSE）

单一 `/ask` 端点，统一走 SSE 逐 token 推送。用户无需等待完整生成，即刻看到内容逐字出现。

### marker + bge 双模型

| 环节 | 模型 | 说明 |
|------|------|------|
| PDF 解析 | marker | 深度模型，文字层 / 扫描件 / 表格 / 公式均输出结构化 Markdown |
| 语义向量 | bge-small-zh-v1.5 | 中文优化，512 维，95MB，轻量高精度 |

### 安全防护

- Bearer Token 鉴权（可开关）
- CORS 白名单 + 安全响应头（nosniff / DENY / XSS-Protection）
- 文件大小前置拦截（>50MB → 413）
- 文档去重（SHA256 内容哈希）
- Prompt 注入防御（system + user prompt 双约束）
- XSS 防护（前端 HTML 转义）

---

## 技术栈

| 层 | 技术 |
|----|------|
| 框架 | FastAPI（异步） |
| 向量库 | ChromaDB（持久化，双集合 children / parents） |
| Embedding | sentence-transformers + bge-small-zh-v1.5（512 维） |
| LLM | DeepSeek / OpenAI 兼容格式（temperature 可调） |
| PDF 解析 | PyMuPDF（毫秒级，生产可升级 marker） |
| Word 解析 | python-docx（表格 / 标题 / 列表结构化提取） |
| 切分 | langchain-text-splitters（RecursiveCharacterTextSplitter） |
| 数据库 | SQLite（WAL 模式，零依赖） |
| 日志 | Python logging（时间戳 + 级别 + 模块名） |

---

## 快速开始

```bash
# 1. 安装依赖（首次需下载 embedding + marker 模型，约 2.5GB）
git clone <repo>
cd rag-api
uv sync

# 2. 配置 LLM API Key
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY / LLM_API_BASE / LLM_MODEL

# 3. 启动
uv run python main.py
# 浏览器打开 http://127.0.0.1:8765
```

---

## 项目结构

```
rag-api/
├── main.py              # FastAPI 应用：路由 + 鉴权 + CORS + 内嵌前端
├── rag_engine.py        # RAG 核心：Parent-Child 切块 + 检索 + LLM 流式
├── pdf_handler.py       # PDF/DOCX 处理器：marker + python-docx
├── database.py          # SQLite 操作层 + 去重
├── config.py            # 配置中心（模型、切块参数、端口等）
├── schemas.py           # Pydantic 请求/响应模型
├── eval_retrieval.py    # 检索命中率评测（30 个用例，4 类场景）
├── pyproject.toml       # uv 依赖声明
├── requirements.txt     # pip 兼容依赖
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 交互界面 |
| GET | `/settings` | 系统配置信息（只读） |
| POST | `/upload` | 上传文档（.pdf / .docx / .txt / .md） |
| POST | `/ask` | 流式问答（SSE，支持 top_k + temperature） |
| GET | `/documents` | 文档列表 |
| DELETE | `/documents/{id}` | 删除单个文档 |
| DELETE | `/documents` | 清空全部文档 |

---

## 前端设置面板

点击右上角 ⚙ 齿轮：

- **Top-K 滑块**（1-20）：控制检索段落数
- **Temperature 滑块**（0-1）：控制回答创造性
- **清空知识库**：一键重置
- **系统信息**：当前 embedding 模型 / LLM 模型 / 切块参数 / 文档数

---

## 检索评测

```bash
PYTHONPATH="" uv run python eval_retrieval.py
```

10 个测试用例，LLM 直接打分，覆盖两个核心维度：

| 指标 | 说明 | 判定方式 |
|------|------|------|
| faithfulness | 回答是否忠于原文（有无编造） | LLM 逐条核对回答中的事实是否在检索段落中有依据 |
| context_recall | 检索是否覆盖了标准答案 | LLM 判断检索段落是否包含标准答案的关键信息 |

> 为什么不直接用 RAGAS？RAGAS 0.2 对 langchain 版本有严格依赖，依赖链脆弱。Demo 场景 60 行自研比 200MB 依赖链更合理。生产环境建议接入 RAGAS 获得完整指标体系。

---

## 面试展示要点

**能展开讲的设计决策：**

1. **为什么 Parent-Child 而不是单层切块？** — 精度 vs 完整性的 trade-off
2. **为什么 marker 而不是 PyMuPDF？** — 表格 / 扫描件 / 公式的结构化提取能力
3. **为什么 bge-small-zh-v1.5 而不是 all-MiniLM？** — 中文语义匹配实测差距
4. **SSE 流式怎么实现的？** — OpenAI stream=True → Python 生成器 → SSE → ReadableStream
5. **安全做了哪些？** — 鉴权 + CORS + 大小限制 + 去重 + Prompt 注入防御 + XSS

**诚实的已知局限：**

- 数值检索依赖纯 embedding（BM25 混合检索可补）
- 小文档场景下边界测试退化为全文返回
- 无多用户体系（生产需 JWT）
- Rate Limit 为 README 标注方案（生产需 Redis）

---

## 生产部署

Docker 一键启动（容器内独立环境，不依赖本地 .venv）：

```bash
docker-compose up --build
```

生产环境建议补充：
- 数据库切换为 PostgreSQL
- Rate Limit 接入 Redis 滑动窗口
- 多用户认证（JWT + RBAC）
- RAGAS 端到端评测（faithfulness / answer_relevancy）
- Prometheus + Grafana 监控

# 🎯 Interview Engine

基于 RAG 和 LLM 的求职辅助系统。上传简历和岗位描述，通过 Web 页面即可进行智能问答、AI 模拟面试、多维度评分和生成面试报告。

## 功能

**智能问答** — 上传文档后自然语言提问，RAG 检索相关内容，LLM 生成回答。

**模拟面试** — 选择简历，系统分析后逐题出题，你作答后给出评分和反馈。面试结束后自动生成报告，包含每题明细、维度分析、关键词命中率等。

**多维度评分** — 每题从技术深度、表达清晰度、逻辑性三个维度评分。

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/your-username/interview-engine.git
cd interview-engine

# 2. 安装依赖
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 3. 配置 API Key
# 复制 .env.example 为 .env，填入你的 LLM_API_KEY
# 支持 DeepSeek / OpenAI 等兼容格式的 API

# 4. 启动服务
.venv\Scripts\python main.py
```

启动后浏览器打开 `http://127.0.0.1:8765` 即可使用。

> 首次启动会自动下载 embedding 模型（~95MB），请保持网络通畅。国内用户会自动使用镜像下载。

## 工作流程

### RAG 问答流程

```mermaid
flowchart TD
    subgraph 入库
        A[上传文档] --> B[双层切块<br>Parent 2000字 / Child 300字]
        B --> C[向量化存入 ChromaDB]
        B --> D[分词建立 BM25 索引]
    end

    subgraph 检索
        E[用户提问] --> F[语义检索 + 关键词检索]
        F --> G[RRF 融合排序]
        G --> H[提取 Parent 级上下文]
    end

    subgraph 生成
        H --> I[LLM 生成回答]
        I --> J[SSE 流式返回]
    end
```

### 面试流程

```mermaid
flowchart TD
    A[选择简历文档] --> B[Analyze LLM<br>分析技能和经验]
    B --> C[Question LLM<br>生成第一道面试题]
    C --> D[用户输入回答]
    D --> E[Score LLM<br>多维度评分 + 反馈]
    E --> F{还有下一题?}
    F -->|是| G[Question LLM<br>基于已答表现出下一题]
    G --> D
    F -->|否| H[Report LLM<br>生成结构化报告]
```

## 项目结构

```
interview-engine/
├── main.py              # FastAPI 应用入口
├── rag_engine.py        # RAG + BM25 混合检索
├── interview_agent.py   # 面试引擎（4 个 LLM 调用）
├── database.py          # SQLite 操作层
├── config.py            # 配置
├── pdf_handler.py       # PDF/DOCX 提取
├── schemas.py           # Pydantic 模型
├── eval_retrieval.py    # 检索评测脚本
└── templates/
    └── index.html       # 前端页面
```

## 技术栈

FastAPI / ChromaDB / sentence-transformers / rank-bm25 / SQLite

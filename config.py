import os
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件（必须在读取任何环境变量之前）

# ========== 日志（替换全项目 print） ==========
import logging
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

# ========== HuggingFace 镜像（国内必备） ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========== Embedding 模型 ==========
EMBEDDING_MODEL = 'BAAI/bge-small-zh-v1.5'  # 中文优化，512维，约95MB

# ========== ChromaDB 持久化路径 ==========
CHROMA_PERSIST_DIR = './chroma_db'

# ========== SQLite 路径 ==========
SQLITE_PATH = os.getenv('SQLITE_PATH', './rag_api.db')

# ========== LLM 配置（兼容 OpenAI API 格式） ==========
# 国内可以用 DeepSeek / Moonshot / 智谱等，它们都兼容 OpenAI 的 API 格式
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_API_BASE = os.getenv('LLM_API_BASE', 'https://api.openai.com/v1')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')

# ========== 服务端口 ==========
# 用不常见的端口避免和 Hermes 冲突（Hermes 内部可能占用 8000）
PORT = int(os.getenv('PORT', '8765'))
HOST = os.getenv('HOST', '127.0.0.1')  # 默认仅本地访问，改为 0.0.0.0 暴露到网络

# ========== 安全配置 ==========
APP_SECRET_KEY = os.getenv('APP_SECRET_KEY', '')  # 空=不开启鉴权（本地开发）
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 上传文件大小上限：50MB

# ========== RAG 参数 ==========
# —— Parent-Child 双层切块 ——
# Child: 小chunk，用于检索匹配（精度高）
# Parent: 大chunk，用于给LLM提供上下文（信息完整）
CHILD_CHUNK_SIZE = 300     # 子chunk大小——检索用，精度优先
CHILD_OVERLAP = 50         # 子chunk重叠
PARENT_CHUNK_SIZE = 2000   # 父chunk大小——给LLM看，上下文完整
PARENT_OVERLAP = 200       # 父chunk重叠
TOP_K = 10                 # 检索返回的子chunk数（越多覆盖越广）

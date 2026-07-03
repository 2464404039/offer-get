FROM python:3.11-slim

WORKDIR /app

# 使用清华镜像加速 Python 包下载
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制项目代码
COPY . .

# 提前下载 Embedding 模型（避免运行时首次请求慢）
# 注意：需要设置 HF_ENDPOINT（在 config.py 中已设置）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

EXPOSE 8765

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]

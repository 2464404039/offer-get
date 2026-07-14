FROM python:3.11-slim

WORKDIR /app

# 国内镜像源加速
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

# 编译依赖（chromadb→grpcio 需要 build-essential）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY . .

# 持久化目录
RUN mkdir -p /app/data

# 非 root 用户运行（安全）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8765

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8765}

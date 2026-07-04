FROM python:3.11-slim

WORKDIR /app

# 国内镜像源
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data

EXPOSE 8765

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8765}

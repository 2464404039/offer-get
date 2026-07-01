from pydantic import BaseModel, Field
from typing import Optional, List


# ========== 请求模型 ==========

class AskRequest(BaseModel):
    """提问请求"""
    query: str                                    # 用户问题
    top_k: int = Field(default=5, ge=1, le=20)   # 检索段落数
    temperature: float = Field(default=0.6, ge=0, le=1)  # LLM 温度


# ========== 响应模型 ==========

class UploadResponse(BaseModel):
    """上传文档的响应"""
    document_id: int                              # 文档ID
    filename: str                                 # 文件名
    chunks: int                                   # 切分后的段落数
    message: str                                  # 提示信息


class SourceItem(BaseModel):
    """检索到的来源段落"""
    document: str                                 # 来源文件名
    content: str                                  # 段落原文
    score: float                                  # 相似度分数


class AskResponse(BaseModel):
    """问答响应"""
    answer: str                                   # LLM 生成的回答
    sources: List[SourceItem]                     # 检索到的参考来源


class DocumentItem(BaseModel):
    """文档列表项"""
    id: int
    filename: str
    created_at: str
    chunk_count: int

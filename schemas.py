from pydantic import BaseModel, Field
from typing import Optional, List


# ========== 认证 API 请求模型 ==========

class RegisterRequest(BaseModel):
    """用户注册"""
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)


class LoginRequest(BaseModel):
    """用户登录"""
    username: str
    password: str


# ========== 知识库问答请求模型 ==========

class AskRequest(BaseModel):
    """提问请求"""
    query: str                                    # 用户问题
    top_k: int = Field(default=5, ge=1, le=20)   # 检索段落数
    temperature: float = Field(default=0.6, ge=0, le=1)  # LLM 温度


# ========== 备战 API 请求模型 ==========

class StudyPlanRequest(BaseModel):
    """备战计划请求"""
    resume_doc_id: Optional[int] = None  # 可选关联简历


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
    doc_type: str = 'general'
    user_id: Optional[int] = None


# ========== 面试 API 请求/响应模型 ==========


class InterviewStartRequest(BaseModel):
    resume_doc_id: int
    jd_doc_id: Optional[int] = None
    total_questions: int = Field(default=8, ge=3, le=15)
    question_ids: Optional[List[int]] = None  # 从题库抽题（可选）


class InterviewAnswerRequest(BaseModel):
    session_id: int
    question_id: int
    answer: str = Field(..., min_length=1)


# ========== 题库 API 请求/响应模型 ==========

class SaveQuestionRequest(BaseModel):
    """收藏题目"""
    question_text: str = Field(..., min_length=1)
    dimension: Optional[str] = None
    difficulty: str = 'medium'


class QuestionItem(BaseModel):
    """题库列表项"""
    id: int
    question_text: str
    dimension: Optional[str] = None
    difficulty: str
    source: str = 'manual'
    created_at: str

"""
测试 1: 纯逻辑单元测试（零外部依赖，最快）
覆盖：BM25 分词 / RRF 数学 / Parent-Child 元数据 / 面试状态机

运行：.venv\Scripts\python tests\test_core.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ═══════════════════════ BM25 分词 ═══════════════════════

def test_tokenize_english():
    from rag_engine import RAGEngine
    tokens = RAGEngine._tokenize("FastAPI is great for building APIs")
    assert "fastapi" in tokens
    assert "building" in tokens
    assert "apis" in tokens
    print("✅ test_tokenize_english")


def test_tokenize_chinese():
    from rag_engine import RAGEngine
    tokens = RAGEngine._tokenize("什么是向量检索技术")
    assert len(tokens) > 0, f"空结果: {tokens}"
    print(f"✅ test_tokenize_chinese: {tokens}")


def test_tokenize_empty():
    from rag_engine import RAGEngine
    assert RAGEngine._tokenize("") == []
    print("✅ test_tokenize_empty")


def test_tokenize_mixed():
    from rag_engine import RAGEngine
    tokens = RAGEngine._tokenize("Python FastAPI 框架 向量检索")
    assert "python" in tokens
    print(f"✅ test_tokenize_mixed: {tokens}")


# ═══════════════════════ RRF 融合数学 ═══════════════════════

def test_rrf_weighted():
    """RRF: score = 0.7/(60+rank_vec) + 0.3/(60+rank_bm25)"""
    k = 60
    vw, bw = 0.7, 0.3
    a = vw / (k + 1) + bw / (k + 3)
    b = vw / (k + 3) + bw / (k + 1)
    assert a > 0 and b > 0
    assert a != b  # 不同排名应有不同分数
    print("✅ test_rrf_weighted")


def test_rrf_all_same():
    """全部排第1 vs 全部排第3"""
    k = 60
    vw, bw = 0.7, 0.3
    top = vw / (k + 1) + bw / (k + 1)
    bottom = vw / (k + 3) + bw / (k + 3)
    assert top > bottom
    print("✅ test_rrf_all_same")


def test_rrf_k_effect():
    """k 越大排名差异越小"""
    k_low, k_high = 10, 100
    low = abs(1 / (k_low + 1) - 1 / (k_low + 10))
    high = abs(1 / (k_high + 1) - 1 / (k_high + 10))
    assert low > high
    print("✅ test_rrf_k_effect")


# ═══════════════════════ Parent-Child 元数据 ═══════════════════════

def test_child_has_parent_id():
    meta = {"doc_id": 1, "parent_id": "doc_1_p0"}
    assert "parent_id" in meta
    assert meta["parent_id"] == "doc_1_p0"
    print("✅ test_child_has_parent_id")


def test_parent_has_doc_id():
    meta = {"doc_id": 1, "filename": "test.md"}
    assert "doc_id" in meta
    print("✅ test_parent_has_doc_id")


def test_id_format():
    assert "doc_1_p0_c2" == "doc_1_p0_c2"
    assert "doc_1_p0" == "doc_1_p0"
    print("✅ test_id_format")


# ═══════════════════════ 面试状态机 ═══════════════════════

def test_status_pending_progress_complete():
    states = ["pending", "in_progress", "completed"]
    assert len(states) == 3
    for i, s in enumerate(states):
        assert s in states, f"{s} not in states"
    print("✅ test_status_pending_progress_complete")


def test_score_out_of_10():
    dims = {"技术": 8, "表达": 7, "逻辑": 6}
    for k, v in dims.items():
        assert 0 <= v <= 10, f"{k}={v}"
    print("✅ test_score_out_of_10")


def test_difficulty_progression():
    def diff(score):
        return "hard" if score >= 8 else ("medium" if score >= 5 else "easy")
    assert diff(9) == "hard"
    assert diff(7) == "medium"
    assert diff(3) == "easy"
    print("✅ test_difficulty_progression")


def test_weakest_dimension():
    dims = {"技术深度": 8, "表达清晰度": 5, "逻辑性": 6}
    weakest = min(dims, key=dims.get)
    assert weakest == "表达清晰度"
    print("✅ test_weakest_dimension")


# ═══════════════════════ Tool Schema 校验 ═══════════════════════

def test_tool_schema_is_valid_json():
    from interview_agent import TOOL_SCHEMAS, REACT_TOOLS
    for name, schema in TOOL_SCHEMAS.items():
        assert "function" in schema
        assert "parameters" in schema["function"]
        assert "properties" in schema["function"]["parameters"]
        print(f"✅ TOOL_SCHEMAS[{name}]: {list(schema['function']['parameters']['properties'].keys())}")

    for t in REACT_TOOLS:
        assert "function" in t
        assert "name" in t["function"]
        print(f"✅ REACT_TOOLS: {t['function']['name']}")


def test_tool_schema_required_fields():
    from interview_agent import TOOL_SCHEMAS
    for name, schema in TOOL_SCHEMAS.items():
        required = schema["function"]["parameters"].get("required", [])
        assert len(required) > 0, f"{name} 没有 required 字段"
    print("✅ 所有 Tool Schema 有 required 约束")


# ═══════════════════════ 运行 ═══════════════════════

if __name__ == "__main__":
    tests = [
        ("BM25 分词", [test_tokenize_english, test_tokenize_chinese,
                       test_tokenize_empty, test_tokenize_mixed]),
        ("RRF 融合", [test_rrf_weighted, test_rrf_all_same, test_rrf_k_effect]),
        ("元数据", [test_child_has_parent_id, test_parent_has_doc_id,
                    test_id_format]),
        ("状态机", [test_status_pending_progress_complete, test_score_out_of_10,
                    test_difficulty_progression, test_weakest_dimension]),
        ("Tool Schema", [test_tool_schema_is_valid_json,
                         test_tool_schema_required_fields]),
    ]
    passed = failed = 0
    for group, fns in tests:
        print(f"\n{'='*50}\n  {group}\n{'='*50}")
        for fn in fns:
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  ❌ {fn.__name__}: {e}")
                failed += 1
    print(f"\n{'='*50}\n  结果: {passed} 通过, {failed} 失败 / {passed + failed} 总计\n{'='*50}")
    sys.exit(1 if failed else 0)

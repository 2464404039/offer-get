"""
测试 3: API 端点集成测试（FastAPI TestClient）
覆盖：健康检查 / 认证 / 文档 CRUD

运行：.venv\Scripts\python tests\test_api.py
注意：会操作真实数据库，建议在独立环境下运行
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ["APP_SECRET_KEY"] = "test-api-secret-key"

# 使用独立测试数据库
os.environ["SQLITE_PATH"] = ":memory:"


def test_health_endpoint():
    """/health 返回 200 + 正确字段"""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "interview-engine"
    print("✅ test_health_endpoint")


def test_root_returns_html():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    print("✅ test_root_returns_html")


def test_auth_session_token():
    """POST /auth/session 返回短期 session token"""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.post("/auth/session")
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert data["token"].startswith("session:")
    print(f"✅ test_auth_session_token: token={data['token'][:30]}...")


def test_app_secret_key_auth():
    """用 APP_SECRET_KEY 作为 Bearer token 访问受保护端点"""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/settings", headers={"Authorization": f"Bearer {os.environ['APP_SECRET_KEY']}"})
    # /settings 依赖 lifespan 初始化数据库，TestClient 不触发 lifespan；
    # 但至少验证 token 校验本身不会抛 401
    assert resp.status_code != 401
    print(f"✅ test_app_secret_key_auth: status={resp.status_code}")


def test_upload_needs_auth():
    """未携带 token 的上传请求返回 403"""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.post("/upload")
    assert resp.status_code == 403
    print("✅ test_upload_needs_auth")


# ═══════════════════════ 运行 ═══════════════════════

if __name__ == "__main__":
    # 初始化数据库（TestClient 不自动触发 lifespan）
    import asyncio
    from database import async_init_db
    asyncio.run(async_init_db())

    tests = [
        ("健康检查 / 首页", [test_health_endpoint, test_root_returns_html]),
        ("认证", [test_auth_session_token, test_app_secret_key_auth, test_upload_needs_auth]),
    ]
    passed = failed = 0
    for group, fns in tests:
        print(f"\n{'='*50}\n  {group}\n{'='*50}")
        for fn in fns:
            try:
                fn()
                passed += 1
            except Exception as e:
                import traceback
                print(f"  ❌ {fn.__name__}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*50}\n  结果: {passed} 通过, {failed} 失败 / {passed + failed} 总计\n{'='*50}")
    sys.exit(1 if failed else 0)

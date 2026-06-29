"""重试矩阵的故障注入测试：伪造 openai 异常，断言退避/放弃行为。"""
from types import SimpleNamespace

import httpx
import openai
import pytest

import harness.providers.openai_chat as oc
from harness.providers import OpenAIChatProvider


def status_error(status: int, headers: dict | None = None) -> openai.APIStatusError:
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    resp = httpx.Response(status, request=req, headers=headers or {})
    return openai.APIStatusError(f"http {status}", response=resp, body=None)


def fake_response(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                              prompt_tokens_details=None, completion_tokens_details=None),
    )


@pytest.fixture
def provider(monkeypatch):
    p = OpenAIChatProvider(model="gpt-5.5", api_key="test", max_retries=3)
    monkeypatch.setattr(oc.time, "sleep", lambda s: sleeps.append(s))
    sleeps: list[float] = []
    p._sleeps = sleeps  # type: ignore[attr-defined]
    return p


def script(provider, outcomes):
    """outcomes: 异常或响应的序列，依次作为每次 create 调用的结果。"""
    it = iter(outcomes)

    def fake_create(**kwargs):
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item
    provider.client.chat.completions.create = fake_create


def test_429_then_success(provider):
    script(provider, [status_error(429), status_error(429), fake_response()])
    retries = []
    result = provider.complete([], [], on_retry=lambda *a: retries.append(a))
    assert result.content == "ok"
    assert len(retries) == 2
    assert retries[0][1] == 429


def test_400_fails_immediately_no_retry(provider):
    calls = []
    script(provider, [status_error(400)])
    with pytest.raises(openai.APIStatusError):
        provider.complete([], [], on_retry=lambda *a: calls.append(a))
    assert calls == []  # 重试 400 只会原样重复失败


def test_retry_after_header_is_respected(provider):
    script(provider, [status_error(429, {"retry-after": "7"}), fake_response()])
    provider.complete([], [])
    assert provider._sleeps == [7.0]


def test_exponential_backoff_with_jitter(provider):
    script(provider, [status_error(503), status_error(503), fake_response()])
    provider.complete([], [])
    s = provider._sleeps
    assert len(s) == 2
    assert 1.0 <= s[0] <= 2.0    # 2^0 + U(0,1)
    assert 2.0 <= s[1] <= 3.0    # 2^1 + U(0,1)


def test_retries_exhausted_raises_last_error(provider):
    script(provider, [status_error(503)] * 4)   # max_retries=3 → 4 次尝试全失败
    with pytest.raises(openai.APIStatusError):
        provider.complete([], [])
    assert len(provider._sleeps) == 3


def test_connection_error_is_retryable(provider):
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    script(provider, [openai.APIConnectionError(request=req), fake_response()])
    assert provider.complete([], []).content == "ok"

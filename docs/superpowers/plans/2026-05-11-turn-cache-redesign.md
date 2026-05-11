# Turn Cache Redesign — Closed-Template Boundary Computation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `_lcp_end`/EOS-scan boundary heuristic with exact closed-template LCP boundaries, simplify the 7-field request surface to 2 fields, and replace the `% budget` chunked-prefill alignment trick with a clean boundary-aware chunk loop.

**Architecture:** `_compute_turn_boundaries` in `batched.py` tokenizes each closed-prefix (no gen-prompt) and LCPs against `full_tokens`, producing `[B_sys, B_1, ..., B_{N-1}]`. The scheduler's mid-prefill callback stores state at each boundary token count in `_boundary_states[B_k]`. The chunk loop checks whether the next unprocessed boundary is within `budget` tokens and shrinks the chunk to land exactly on it.

**Tech Stack:** Python, mlx-lm `BatchGenerator`, HuggingFace tokenizer `apply_chat_template`, pytest (no real model — tokenizer-only tests).

---

## File Map

| File | Change |
|---|---|
| `vllm_mlx/engine/batched.py` | Replace `_compute_prefix_boundary` → `_compute_turn_boundaries`; update callers in `chat`, `stream_chat`, `generate`, `stream_generate` |
| `vllm_mlx/request.py` | Remove `prefix_boundary`, `sys_end_boundary`, `turn_boundaries`; add `_turn_boundaries: List[int]`, `_boundary_states: Dict[int, Any]` |
| `vllm_mlx/engine_core.py` | Update `add_request` signature: remove three params, add `turn_boundaries: list[int]` |
| `vllm_mlx/scheduler.py` | (a) `_messages_to_segments` — read `_turn_boundaries`; (b) mid-prefill callback — write `_boundary_states`; (c) chunk-loop init + continuation — boundary-aware sizing; (d) `_cleanup_finished` store loop — read `_boundary_states` |
| `tests/test_turn_prefix_cache.py` | Update tests using old fields; add `_compute_turn_boundaries` unit tests |

---

## Task 1: `_compute_turn_boundaries` in `batched.py`

**Files:**
- Modify: `vllm_mlx/engine/batched.py` (replace `_compute_prefix_boundary` around line 985)
- Test: `tests/test_turn_prefix_cache.py` (new tests at bottom)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_turn_prefix_cache.py`:

```python
# ── _compute_turn_boundaries tests (tokenizer-only, no model) ──────────────

class _MockTok:
    """Char-level tokenizer with a minimal chat template."""
    unk_token_id = None

    def encode(self, text):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<{role}>{content}</{role}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


def _make_engine_with_mock_tok():
    """Return a BatchedEngine stub with the mock tokenizer wired in."""
    from vllm_mlx.engine.batched import BatchedEngine
    eng = object.__new__(BatchedEngine)
    eng._is_mllm = False
    eng._tokenizer = _MockTok()
    eng._processor = None
    eng._model_name = "mock"
    return eng


def test_compute_turn_boundaries_single_turn():
    """First turn (no completed assistant turn): returns [B_sys]."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "HELLO"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    assert len(boundaries) == 1, f"Expected 1 boundary, got {boundaries}"
    # B_sys should be the byte-length of "<system>SYS</system>"
    expected_sys_text = "<system>SYS</system>"
    assert boundaries[0] == len(expected_sys_text.encode("utf-8")), (
        f"B_sys={boundaries[0]}, expected {len(expected_sys_text.encode('utf-8'))}"
    )


def test_compute_turn_boundaries_two_turns():
    """Two completed turns returns [B_sys, B_1, B_2]."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "U3"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    assert len(boundaries) == 3, f"Expected 3 boundaries, got {boundaries}"
    assert boundaries[0] < boundaries[1] < boundaries[2], (
        f"Boundaries not strictly increasing: {boundaries}"
    )


def test_compute_turn_boundaries_no_system():
    """No system message → returns []."""
    eng = _make_engine_with_mock_tok()
    messages = [{"role": "user", "content": "HI"}]
    assert eng._compute_turn_boundaries(messages) == []


def test_compute_turn_boundaries_exact_position():
    """B_sys is the exact LCP of closed [sys] template against full_tokens."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    # Reconstruct what B_1 should be:
    # full = <system>SYS</system><user>U1</user><assistant>A1</assistant><assistant>
    # closed [sys,u1,a1] = <system>SYS</system><user>U1</user><assistant>A1</assistant>
    tok = _MockTok()
    closed_text = "<system>SYS</system><user>U1</user><assistant>A1</assistant>"
    expected_b1 = len(closed_text.encode("utf-8"))
    assert boundaries[1] == expected_b1, (
        f"B_1={boundaries[1]}, expected {expected_b1}"
    )


def test_compute_turn_boundaries_strictly_increasing():
    """Every boundary in the returned list is strictly greater than the previous."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "U3"},
        {"role": "assistant", "content": "A3"},
        {"role": "user", "content": "U4"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    assert len(boundaries) == 4  # B_sys + 3 completed turns
    for i in range(len(boundaries) - 1):
        assert boundaries[i] < boundaries[i + 1], (
            f"boundaries[{i}]={boundaries[i]} >= boundaries[{i+1}]={boundaries[i+1]}"
        )


def test_compute_turn_boundaries_last_boundary_less_than_full():
    """The last boundary must be < len(full_tokens) (user segment exists after it)."""
    eng = _make_engine_with_mock_tok()
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "Q"},
    ]
    boundaries = eng._compute_turn_boundaries(messages)
    tok = _MockTok()
    full_tokens = tok.encode(tok.apply_chat_template(messages, add_generation_prompt=True))
    assert boundaries[-1] < len(full_tokens), (
        f"Last boundary {boundaries[-1]} >= full_tokens length {len(full_tokens)}"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/tibo/Projects/vllm-mlx
python -m pytest tests/test_turn_prefix_cache.py::test_compute_turn_boundaries_single_turn -xvs 2>&1 | tail -20
```

Expected: `FAILED` or `AttributeError: '_compute_turn_boundaries'`

- [ ] **Step 3: Implement `_compute_turn_boundaries` in `batched.py`**

In `vllm_mlx/engine/batched.py`, add the new method right after the class definition of `_compute_prefix_boundary` (around line 985) and update the two callers (`chat` and `stream_chat`).

Replace the `_compute_prefix_boundary` method (lines 985–1105) with:

```python
def _compute_turn_boundaries(
    self,
    messages: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Compute exact token boundaries for cache segmentation using closed-template LCP.

    Returns [B_sys, B_1, ..., B_{N-1}] where B_k is the token position just after
    the k-th completed assistant turn (via LCP of closed-template prefix against
    full_tokens). Returns [] if no system message.

    Note: `tools` is intentionally absent — tool schemas are embedded in the
    system prompt in this codebase.
    """
    if not messages or messages[0].get("role") != "system":
        return []

    tokenizer = self.tokenizer
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    if not hasattr(tokenizer, "apply_chat_template"):
        return []

    try:
        full_prompt = self._apply_chat_template(
            messages, chat_template_kwargs=chat_template_kwargs
        )
        full_tokens = tokenizer.encode(full_prompt)
        if not full_tokens:
            return []

        def _lcp_closed(prefix_messages: list[dict]) -> int:
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": False,
            }
            if chat_template_kwargs:
                for k, v in chat_template_kwargs.items():
                    if k != "add_generation_prompt":
                        kwargs[k] = v
            try:
                prefix_text = tokenizer.apply_chat_template(prefix_messages, **kwargs)
            except TypeError:
                try:
                    prefix_text = tokenizer.apply_chat_template(
                        prefix_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    return 0
            except Exception:
                return 0
            prefix_tokens = tokenizer.encode(prefix_text)
            lcp = 0
            for j in range(min(len(full_tokens), len(prefix_tokens))):
                if full_tokens[j] != prefix_tokens[j]:
                    break
                lcp = j + 1
            return lcp

        boundaries: list[int] = []

        B_sys = _lcp_closed([messages[0]])
        if B_sys <= 0 or B_sys >= len(full_tokens):
            return []
        boundaries.append(B_sys)

        k = 1
        while 2 * k + 1 <= len(messages):
            prefix = messages[: 2 * k + 1]
            if prefix[-1].get("role") != "assistant":
                break
            B_k = _lcp_closed(prefix)
            if B_k <= boundaries[-1] or B_k >= len(full_tokens):
                break
            boundaries.append(B_k)
            k += 1

        return boundaries
    except Exception:
        return []
```

In `chat` (around line 962) replace:

```python
        prefix_boundary, sys_end_boundary, turn_boundaries = self._compute_prefix_boundary(
            messages,
            tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        if prefix_boundary > 0:
            kwargs["prefix_boundary"] = prefix_boundary
        if sys_end_boundary > 0:
            kwargs["sys_end_boundary"] = sys_end_boundary
        if turn_boundaries:
            kwargs["turn_boundaries"] = turn_boundaries
```

with:

```python
        turn_boundaries = self._compute_turn_boundaries(
            messages,
            chat_template_kwargs=chat_template_kwargs,
        )
        if turn_boundaries:
            kwargs["turn_boundaries"] = turn_boundaries
```

In `stream_chat` (around line 1168) replace the equivalent block:

```python
        prefix_boundary, sys_end_boundary, turn_boundaries = self._compute_prefix_boundary(
            messages,
            tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        if prefix_boundary > 0:
            kwargs["prefix_boundary"] = prefix_boundary
        if sys_end_boundary > 0:
            kwargs["sys_end_boundary"] = sys_end_boundary
        if turn_boundaries:
            kwargs["turn_boundaries"] = turn_boundaries
```

with:

```python
        turn_boundaries = self._compute_turn_boundaries(
            messages,
            chat_template_kwargs=chat_template_kwargs,
        )
        if turn_boundaries:
            kwargs["turn_boundaries"] = turn_boundaries
```

In `generate` (around line 782) and `stream_generate` (around line 881), replace:

```python
        prefix_boundary = kwargs.pop("prefix_boundary", 0)
        sys_end_boundary = kwargs.pop("sys_end_boundary", 0)
        turn_boundaries = kwargs.pop("turn_boundaries", [])
        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            prefix_boundary=prefix_boundary,
            sys_end_boundary=sys_end_boundary,
            turn_boundaries=turn_boundaries,
        )
```

with:

```python
        turn_boundaries = kwargs.pop("turn_boundaries", [])
        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            turn_boundaries=turn_boundaries,
        )
```

And for `stream_generate`:

```python
        prefix_boundary = kwargs.pop("prefix_boundary", 0)
        sys_end_boundary = kwargs.pop("sys_end_boundary", 0)
        turn_boundaries = kwargs.pop("turn_boundaries", [])
        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            prefix_boundary=prefix_boundary,
            sys_end_boundary=sys_end_boundary,
            turn_boundaries=turn_boundaries,
        )
```

with:

```python
        turn_boundaries = kwargs.pop("turn_boundaries", [])
        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            turn_boundaries=turn_boundaries,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_compute_turn_boundaries_single_turn tests/test_turn_prefix_cache.py::test_compute_turn_boundaries_two_turns tests/test_turn_prefix_cache.py::test_compute_turn_boundaries_no_system tests/test_turn_prefix_cache.py::test_compute_turn_boundaries_exact_position tests/test_turn_prefix_cache.py::test_compute_turn_boundaries_strictly_increasing tests/test_turn_prefix_cache.py::test_compute_turn_boundaries_last_boundary_less_than_full -xvs 2>&1 | tail -30
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/engine/batched.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn_cache): replace _compute_prefix_boundary with _compute_turn_boundaries"
```

---

## Task 2: Update `Request` dataclass fields

**Files:**
- Modify: `vllm_mlx/request.py` (lines 119–121)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_request_has_turn_boundaries_field():
    """Request accepts _turn_boundaries and _boundary_states without error."""
    from vllm_mlx.request import Request, SamplingParams
    req = Request(
        request_id="r1",
        prompt="hi",
        sampling_params=SamplingParams(),
    )
    assert req._turn_boundaries == []
    assert req._boundary_states == {}


def test_request_turn_boundaries_field_set():
    """_turn_boundaries can be set at construction and read back."""
    from vllm_mlx.request import Request, SamplingParams
    req = Request(
        request_id="r1",
        prompt="hi",
        sampling_params=SamplingParams(),
        _turn_boundaries=[10, 40, 85],
    )
    assert req._turn_boundaries == [10, 40, 85]


def test_request_no_old_boundary_fields():
    """Request does NOT have prefix_boundary, sys_end_boundary, or turn_boundaries."""
    from vllm_mlx.request import Request, SamplingParams
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(Request)}
    assert "prefix_boundary" not in field_names
    assert "sys_end_boundary" not in field_names
    assert "turn_boundaries" not in field_names
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_request_no_old_boundary_fields -xvs 2>&1 | tail -10
```

Expected: FAILED — `prefix_boundary` still in fields.

- [ ] **Step 3: Update `request.py`**

In `vllm_mlx/request.py`, replace lines 119–121:

```python
    prefix_boundary: int = 0  # Token count for shared prefix (messages[:-1])
    sys_end_boundary: int = 0  # Token count at end of system prompt only (stable across turns)
    turn_boundaries: List[int] = field(default_factory=list)  # Token positions before each intermediate user message
```

with:

```python
    _turn_boundaries: List[int] = field(default_factory=list)  # [B_sys, B_1, ..., B_{N-1}]
    _boundary_states: Dict[int, Any] = field(default_factory=dict)  # boundary → extracted KV state
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_request_has_turn_boundaries_field tests/test_turn_prefix_cache.py::test_request_turn_boundaries_field_set tests/test_turn_prefix_cache.py::test_request_no_old_boundary_fields -xvs 2>&1 | tail -20
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/request.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn_cache): replace 3 boundary fields with _turn_boundaries + _boundary_states on Request"
```

---

## Task 3: Update `engine_core.py` `add_request` signature

**Files:**
- Modify: `vllm_mlx/engine_core.py` (around line 339)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_engine_core_add_request_accepts_turn_boundaries():
    """add_request accepts turn_boundaries (list) and sets _turn_boundaries on Request."""
    from unittest.mock import MagicMock, AsyncMock
    import asyncio
    from vllm_mlx.engine_core import AsyncEngineCore

    core = object.__new__(AsyncEngineCore)
    core.config = MagicMock()
    core.config.stream_interval = 1
    core.scheduler = MagicMock()
    core.scheduler.add_request = MagicMock()
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}

    loop = asyncio.new_event_loop()
    try:
        request_id = loop.run_until_complete(
            core.add_request(
                prompt="hello",
                turn_boundaries=[10, 30],
            )
        )
    finally:
        loop.close()

    assert core.scheduler.add_request.called
    added_req = core.scheduler.add_request.call_args[0][0]
    assert added_req._turn_boundaries == [10, 30]
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_engine_core_add_request_accepts_turn_boundaries -xvs 2>&1 | tail -15
```

Expected: FAILED — old signature has `prefix_boundary`, `sys_end_boundary`, `turn_boundaries` (old meaning).

- [ ] **Step 3: Update `engine_core.py`**

In `vllm_mlx/engine_core.py`, replace the `add_request` method signature and body (lines 339–393):

Replace parameters:
```python
        prefix_boundary: int = 0,
        sys_end_boundary: int = 0,
        turn_boundaries: Optional[List[int]] = None,
```

with:
```python
        turn_boundaries: Optional[List[int]] = None,
```

Remove from the docstring lines:
```
            prefix_boundary: Token count before last user message (dynamic per turn)
            sys_end_boundary: Token count at end of system prompt (stable across turns)
            turn_boundaries: Token positions before each intermediate user message
```

Replace with:
```
            turn_boundaries: [B_sys, B_1, ..., B_{N-1}] boundary token positions
```

Replace in the `Request(...)` constructor:
```python
            prefix_boundary=prefix_boundary,
            sys_end_boundary=sys_end_boundary,
            turn_boundaries=turn_boundaries or [],
```

with:
```python
            _turn_boundaries=turn_boundaries or [],
```

Also find the `generate` method in `AsyncEngineCore` (around line 784) and do the same parameter surgery — remove `prefix_boundary` and `sys_end_boundary`, keep `turn_boundaries`.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_engine_core_add_request_accepts_turn_boundaries -xvs 2>&1 | tail -15
```

Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/engine_core.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn_cache): update engine_core.add_request to accept turn_boundaries list"
```

---

## Task 4: Update `_messages_to_segments` in `scheduler.py`

**Files:**
- Modify: `vllm_mlx/scheduler.py` (`_messages_to_segments`, lines 3420–3472)

- [ ] **Step 1: Write the failing tests**

In `tests/test_turn_prefix_cache.py`, replace the two helpers `_make_multi_turn_request` and the tests that use `prefix_boundary`/`sys_end_boundary` with new versions using `_turn_boundaries`. Add these new tests alongside the old ones (the old ones will break and can be deleted once the new ones pass):

```python
def _make_request_with_boundaries(prompt_token_ids, turn_boundaries):
    """Create a MagicMock request with _turn_boundaries."""
    from unittest.mock import MagicMock
    req = MagicMock()
    req.prompt_token_ids = prompt_token_ids
    req._turn_boundaries = turn_boundaries
    return req


def test_messages_to_segments_new_single_turn():
    """Single-turn: [sys] + [user] segments."""
    from vllm_mlx.scheduler import Scheduler
    sched = object.__new__(Scheduler)

    sys_toks = list(range(10))
    user_toks = [100, 101]
    all_toks = sys_toks + user_toks
    B_sys = len(sys_toks)

    req = _make_request_with_boundaries(all_toks, [B_sys])
    segs = sched._messages_to_segments(req)

    assert len(segs) == 2
    assert segs[0].role == "system"
    assert segs[0].token_ids == sys_toks
    assert segs[1].role == "user"
    assert segs[1].token_ids == user_toks


def test_messages_to_segments_new_two_boundaries():
    """Two turns: [sys] + [conv] + [user] segments."""
    from vllm_mlx.scheduler import Scheduler
    sched = object.__new__(Scheduler)

    sys_toks = list(range(10))
    conv_toks = list(range(10, 30))   # u1+a1
    user_toks = [100, 101]
    all_toks = sys_toks + conv_toks + user_toks
    B_sys = len(sys_toks)
    B_1 = len(sys_toks) + len(conv_toks)

    req = _make_request_with_boundaries(all_toks, [B_sys, B_1])
    segs = sched._messages_to_segments(req)

    assert len(segs) == 3
    assert segs[0].role == "system"
    assert segs[0].token_ids == sys_toks
    assert segs[1].role == "conversation"
    assert segs[1].token_ids == conv_toks
    assert segs[2].role == "user"
    assert segs[2].token_ids == user_toks


def test_messages_to_segments_new_no_boundaries():
    """No boundaries → []."""
    from vllm_mlx.scheduler import Scheduler
    sched = object.__new__(Scheduler)
    req = _make_request_with_boundaries(list(range(20)), [])
    segs = sched._messages_to_segments(req)
    assert segs == []


def test_messages_to_segments_new_boundary_at_end():
    """B_sys at end of full tokens → [] (no user segment)."""
    from vllm_mlx.scheduler import Scheduler
    sched = object.__new__(Scheduler)
    all_toks = list(range(10))
    req = _make_request_with_boundaries(all_toks, [len(all_toks)])
    segs = sched._messages_to_segments(req)
    assert segs == []


def test_messages_to_segments_new_three_boundaries():
    """Three turns: [sys] + [conv1] + [conv2] + [user] segments."""
    from vllm_mlx.scheduler import Scheduler
    sched = object.__new__(Scheduler)

    sys_toks = list(range(5))
    conv1_toks = list(range(5, 15))
    conv2_toks = list(range(15, 28))
    user_toks = [200, 201, 202]
    all_toks = sys_toks + conv1_toks + conv2_toks + user_toks
    B_sys = 5
    B_1 = 15
    B_2 = 28

    req = _make_request_with_boundaries(all_toks, [B_sys, B_1, B_2])
    segs = sched._messages_to_segments(req)

    assert len(segs) == 4
    assert [s.role for s in segs] == ["system", "conversation", "conversation", "user"]
    assert segs[0].token_ids == sys_toks
    assert segs[1].token_ids == conv1_toks
    assert segs[2].token_ids == conv2_toks
    assert segs[3].token_ids == user_toks


def test_messages_to_segments_new_sys_stable():
    """Sys segment has identical token_ids regardless of how many turns follow."""
    from vllm_mlx.scheduler import Scheduler
    sched = object.__new__(Scheduler)

    sys_toks = list(range(50))
    B_sys = 50
    u1 = list(range(50, 60)); a1 = list(range(100, 110))
    u2 = list(range(200, 205)); a2 = list(range(300, 308))
    u3 = list(range(400, 403))

    # Turn 1 request
    req1 = _make_request_with_boundaries(sys_toks + u1, [B_sys])
    # Turn 2 request
    B_1 = B_sys + len(u1) + len(a1)
    req2 = _make_request_with_boundaries(sys_toks + u1 + a1 + u2, [B_sys, B_1])
    # Turn 3 request
    B_2 = B_1 + len(u2) + len(a2)
    req3 = _make_request_with_boundaries(sys_toks + u1 + a1 + u2 + a2 + u3, [B_sys, B_1, B_2])

    segs1 = sched._messages_to_segments(req1)
    segs2 = sched._messages_to_segments(req2)
    segs3 = sched._messages_to_segments(req3)

    assert segs1[0].token_ids == sys_toks
    assert segs2[0].token_ids == sys_toks
    assert segs3[0].token_ids == sys_toks
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_messages_to_segments_new_single_turn -xvs 2>&1 | tail -15
```

Expected: FAILED — `_messages_to_segments` reads `prefix_boundary`/`sys_end_boundary`, not `_turn_boundaries`.

- [ ] **Step 3: Replace `_messages_to_segments` in `scheduler.py`**

Replace the method body (lines 3420–3472) with:

```python
    def _messages_to_segments(self, request: "Request") -> list:
        """Split a request's token sequence into per-message Segment objects.

        Uses _turn_boundaries = [B_sys, B_1, ..., B_{N-1}] computed by
        _compute_turn_boundaries in batched.py.

        Produces:
          1. Segment(role="system",       token_ids=full[0      : B_sys ])
          2. Segment(role="conversation", token_ids=full[B_sys  : B_1  ])  (if N >= 2)
          ...
          k. Segment(role="conversation", token_ids=full[B_{k-2}: B_{k-1}])
          k+1. Segment(role="user",       token_ids=full[B_{N-1}:       ])
        """
        from .turn_prefix_cache import Segment

        full_tokens = list(request.prompt_token_ids or [])
        if not full_tokens:
            return []

        _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
        if not _turn_boundaries:
            return []

        B_sys = _turn_boundaries[0]
        if B_sys <= 0 or B_sys >= len(full_tokens):
            return []

        segments: list[Segment] = [
            Segment(role="system", token_ids=full_tokens[:B_sys])
        ]

        prev = B_sys
        for B_k in _turn_boundaries[1:]:
            if B_k > prev and B_k < len(full_tokens):
                segments.append(Segment(role="conversation", token_ids=full_tokens[prev:B_k]))
                prev = B_k

        if prev < len(full_tokens):
            segments.append(Segment(role="user", token_ids=full_tokens[prev:]))

        return segments if len(segments) > 1 else []
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_turn_prefix_cache.py -k "messages_to_segments_new" -xvs 2>&1 | tail -20
```

Expected: all 6 tests PASSED

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
python -m pytest tests/test_turn_prefix_cache.py -x 2>&1 | tail -30
```

The old tests that use `prefix_boundary`/`sys_end_boundary` will now fail. That is expected; they are addressed in Task 8.

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn_cache): _messages_to_segments reads _turn_boundaries"
```

---

## Task 5: Update mid-prefill save callback in `scheduler.py`

**Files:**
- Modify: `vllm_mlx/scheduler.py` (`_make_mid_prefill_save_callback`, lines 1529–1634)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_turn_prefix_cache.py`:

```python
def _make_minimal_scheduler_new():
    """Minimal scheduler for testing new mid-prefill callback."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig
    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.turn_cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.requests = {}
    sched.uid_to_request_id = {}
    return sched


def test_mid_prefill_saves_boundary_state():
    """_mid_prefill_save stores state in _boundary_states[B] when at a boundary."""
    from unittest.mock import MagicMock
    sched = _make_minimal_scheduler_new()

    req = MagicMock()
    req.prompt_token_ids = list(range(15))
    req._turn_boundaries = [10]   # B_sys=10
    req.cached_tokens = 0
    req._boundary_states = {}
    sched.requests["r1"] = req
    sched.uid_to_request_id[1] = "r1"

    mock_cache = [_MockKVLayer(10), _MockKVLayer(10)]
    cb = sched._make_mid_prefill_save_callback(save_interval=8192)
    cb(uid=1, processed_tokens=10, prompt_cache=mock_cache)

    assert 10 in req._boundary_states, "_boundary_states[10] not set"
    assert isinstance(req._boundary_states[10], list)
    assert len(req._boundary_states[10]) == 2


def test_mid_prefill_does_not_save_away_from_boundary():
    """_mid_prefill_save does NOT store state when not at a boundary."""
    from unittest.mock import MagicMock
    sched = _make_minimal_scheduler_new()

    req = MagicMock()
    req.prompt_token_ids = list(range(20))
    req._turn_boundaries = [10]
    req.cached_tokens = 0
    req._boundary_states = {}
    sched.requests["r1"] = req
    sched.uid_to_request_id[1] = "r1"

    mock_cache = [_MockKVLayer(5)]
    cb = sched._make_mid_prefill_save_callback(save_interval=8192)
    cb(uid=1, processed_tokens=5, prompt_cache=mock_cache)

    assert req._boundary_states == {}


def test_mid_prefill_saves_multiple_boundaries():
    """_mid_prefill_save correctly handles multiple boundaries."""
    from unittest.mock import MagicMock
    sched = _make_minimal_scheduler_new()

    req = MagicMock()
    req.prompt_token_ids = list(range(40))
    req._turn_boundaries = [10, 25]   # B_sys=10, B_1=25
    req.cached_tokens = 0
    req._boundary_states = {}
    sched.requests["r1"] = req
    sched.uid_to_request_id[1] = "r1"

    cb = sched._make_mid_prefill_save_callback(save_interval=8192)

    cb(uid=1, processed_tokens=10, prompt_cache=[_MockKVLayer(10)])
    assert 10 in req._boundary_states

    cb(uid=1, processed_tokens=25, prompt_cache=[_MockKVLayer(25)])
    assert 25 in req._boundary_states

    cb(uid=1, processed_tokens=15, prompt_cache=[_MockKVLayer(15)])
    assert 15 not in req._boundary_states  # not a boundary
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_mid_prefill_saves_boundary_state -xvs 2>&1 | tail -15
```

Expected: FAILED — callback reads `prefix_boundary`/`sys_end_boundary`, not `_turn_boundaries`.

- [ ] **Step 3: Replace the turn_cache branch in `_make_mid_prefill_save_callback`**

In `scheduler.py`, the callback function `_mid_prefill_save` (lines 1541–1634).

Replace the section starting at line 1549 (everything in the `_mid_prefill_save` closure):

```python
        def _mid_prefill_save(uid, processed_tokens, prompt_cache):
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                return
            request = self.requests.get(request_id)
            if not request or not request.prompt_token_ids:
                return

            total_processed = (request.cached_tokens or 0) + processed_tokens

            # memory_aware_cache branch: unchanged throttle logic
            prefix_boundary = getattr(request, "prefix_boundary", 0)
            _tb_old = getattr(request, "turn_boundaries", None)
            turn_boundaries_old = _tb_old if isinstance(_tb_old, list) else []
            sys_end_boundary = getattr(request, "sys_end_boundary", 0) or prefix_boundary

            at_sys_end = sys_end_boundary > 0 and total_processed == sys_end_boundary
            at_prefix_boundary = prefix_boundary > 0 and total_processed == prefix_boundary
            at_turn_boundary_idx = next(
                (i for i, b in enumerate(turn_boundaries_old) if b > 0 and total_processed == b),
                -1,
            )
            at_any_boundary_old = at_sys_end or at_prefix_boundary or at_turn_boundary_idx >= 0

            last_save = getattr(request, "_mid_prefill_last_save", 0)
            if self.memory_aware_cache is not None:
                if not at_any_boundary_old and total_processed - last_save < save_interval:
                    return
                extracted = self._extract_cache_states(prompt_cache)
                if extracted:
                    reconstructed = self._reconstruct_cache_from_states(extracted)
                    if reconstructed:
                        prefix_tokens = list(request.prompt_token_ids[:total_processed])
                        old_key = getattr(request, "_mid_prefill_cache_key", None)
                        if old_key is not None:
                            self.memory_aware_cache.remove(list(old_key))
                        import time as _time
                        _t0 = _time.monotonic()
                        stored = self.memory_aware_cache.store(prefix_tokens, reconstructed)
                        _dt = _time.monotonic() - _t0
                        if stored:
                            request._mid_prefill_last_save = total_processed
                            request._mid_prefill_cache_key = tuple(prefix_tokens)
                            logger.info(
                                f"[mid_prefill_cache] request={request_id[:12]} "
                                f"saved {total_processed}/{len(request.prompt_token_ids)} tokens "
                                f"({total_processed * 100 // len(request.prompt_token_ids)}%) "
                                f"store_time={_dt:.3f}s"
                            )

            # turn_cache branch: store state at each boundary in _boundary_states
            if self.turn_cache is not None:
                _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
                if total_processed in _turn_boundaries:
                    extracted = self._extract_cache_states(prompt_cache)
                    if extracted:
                        if not hasattr(request, "_boundary_states") or request._boundary_states is None:
                            request._boundary_states = {}
                        request._boundary_states[total_processed] = extracted
                        logger.info(
                            f"[turn_cache] boundary_state captured at {total_processed} "
                            f"layers={len(extracted)} for {request_id[:12]}"
                        )
```

(The `return _mid_prefill_save` line at the end remains unchanged.)

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_turn_prefix_cache.py -k "mid_prefill_saves" -xvs 2>&1 | tail -20
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn_cache): mid-prefill callback uses _boundary_states dict"
```

---

## Task 6: Update `_cleanup_finished` store loop in `scheduler.py`

**Files:**
- Modify: `vllm_mlx/scheduler.py` (`_cleanup_finished`, lines 2596–2680)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_store_side_uses_boundary_states():
    """After generation, nodes get state from _boundary_states[B_k]."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(10))
    user_tokens = list(range(10, 15))
    B_sys = 10

    sys_state = _make_extracted_state(n_layers=2, n_tokens=10)
    req = MagicMock()
    req.prompt_token_ids = sys_tokens + user_tokens
    req._turn_boundaries = [B_sys]
    req._boundary_states = {B_sys: sys_state}
    req._extracted_cache = _make_extracted_state(n_layers=2, n_tokens=15)
    req._turn_cache_path = []
    req.output_token_ids = [500, 501]

    segments = sched._messages_to_segments(req)
    assert len(segments) == 2

    parent = cache.root
    matched_depth = 0
    new_segments = segments
    _turn_boundaries = req._turn_boundaries
    _boundary_states = req._boundary_states
    parent_before_user = parent

    for i, segment in enumerate(new_segments):
        abs_idx = matched_depth + i
        is_sys = segment.role == "system" and abs_idx == 0
        is_last = i == len(new_segments) - 1

        if is_last:
            state = None
        elif abs_idx < len(_turn_boundaries):
            state = _boundary_states.get(_turn_boundaries[abs_idx])
        else:
            state = None

        parent = cache.insert(parent, segment, [], [], state, is_system_prompt=is_sys)
        if not is_last:
            parent_before_user = parent

    sys_node = list(cache.root.children.values())[0]
    assert sys_node.recurrent_state is not None
    assert sys_node.recurrent_state is sys_state

    user_node = list(sys_node.children.values())[0]
    assert user_node.recurrent_state is None  # user = structural


def test_store_side_conv_node_gets_boundary_state():
    """Conv node gets _boundary_states[B_1], not sys state."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(10))
    conv_tokens = list(range(10, 25))
    user_tokens = [200, 201]
    B_sys = 10
    B_1 = 25

    sys_state = _make_extracted_state(n_layers=1, n_tokens=10)
    conv_state = _make_extracted_state(n_layers=1, n_tokens=25)

    req = MagicMock()
    req.prompt_token_ids = sys_tokens + conv_tokens + user_tokens
    req._turn_boundaries = [B_sys, B_1]
    req._boundary_states = {B_sys: sys_state, B_1: conv_state}
    req._extracted_cache = _make_extracted_state(n_layers=1, n_tokens=27)
    req._turn_cache_path = []
    req.output_token_ids = [999]

    segments = sched._messages_to_segments(req)
    assert len(segments) == 3

    parent = cache.root
    _turn_boundaries = req._turn_boundaries
    _boundary_states = req._boundary_states
    parent_before_user = parent

    for i, segment in enumerate(segments):
        is_sys = segment.role == "system" and i == 0
        is_last = i == len(segments) - 1
        if is_last:
            state = None
        elif i < len(_turn_boundaries):
            state = _boundary_states.get(_turn_boundaries[i])
        else:
            state = None
        parent = cache.insert(parent, segment, [], [], state, is_system_prompt=is_sys)
        if not is_last:
            parent_before_user = parent

    sys_node = list(cache.root.children.values())[0]
    conv_node = list(sys_node.children.values())[0]
    assert conv_node.recurrent_state is conv_state
```

- [ ] **Step 2: Run to confirm tests pass with inline logic**

```bash
python -m pytest tests/test_turn_prefix_cache.py::test_store_side_uses_boundary_states tests/test_turn_prefix_cache.py::test_store_side_conv_node_gets_boundary_state -xvs 2>&1 | tail -20
```

These tests use inline store logic that mirrors the new implementation, so they should PASS already. If not, debug before touching `scheduler.py`.

- [ ] **Step 3: Update `_cleanup_finished` turn_cache branch in `scheduler.py`**

In `scheduler.py`, find the turn_cache store branch inside `_cleanup_finished` (lines 2596–2680). Replace:

```python
                elif self.turn_cache is not None:
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            segments = self._messages_to_segments(request)
                            path = getattr(request, "_turn_cache_path", [])
                            matched_depth = len(path)
                            parent = path[-1] if path else self.turn_cache.root
                            new_segments = segments[matched_depth:]
                            _tb = getattr(request, "turn_boundaries", None)
                            turn_boundaries = _tb if isinstance(_tb, list) else []
                            turn_boundary_states = getattr(request, "_turn_boundary_states", {}) or {}
                            # parent_before_user tracks the node that will be the parent
                            # of both the user structural node AND the response node.
                            parent_before_user = parent
                            for i, segment in enumerate(new_segments):
                                abs_idx = matched_depth + i
                                is_sys = segment.role == "system" and abs_idx == 0
                                is_last = i == len(new_segments) - 1
                                if is_sys:
                                    state = getattr(request, "_sys_prompt_state", None)
                                elif is_last:
                                    state = None
                                elif segment.role == "conversation":
                                    h = abs_idx - 1
                                    if h < len(turn_boundaries):
                                        state = turn_boundary_states.get(h)
                                    else:
                                        state = getattr(request, "_conv_end_state", None)
                                else:
                                    state = None
                                parent = self.turn_cache.insert(
                                    parent, segment, [], [], state, is_system_prompt=is_sys
                                )
                                if not is_last:
                                    parent_before_user = parent

                            # Store the completed exchange (user_tokens + output_tokens) as a
                            # conversation node under the same parent as the user node.
                            prefix_boundary = getattr(request, "prefix_boundary", 0)
                            if (
                                request.output_token_ids
                                and prefix_boundary > 0
                                and request.prompt_token_ids
                                and new_segments
                            ):
                                from .turn_prefix_cache import Segment as _Seg
                                response_tokens = (
                                    list(request.prompt_token_ids[prefix_boundary:])
                                    + list(request.output_token_ids)
                                )
                                ec = request._extracted_cache
                                if isinstance(ec, list) and ec and isinstance(ec[0], dict):
                                    resp_state = ec
                                else:
                                    resp_state = self._extract_cache_states(ec) or None
                                self.turn_cache.insert(
                                    parent_before_user,
                                    _Seg(role="conversation", token_ids=response_tokens),
                                    [], [], resp_state, is_system_prompt=False,
                                )
                                logger.info(
                                    f"[turn_cache] stored response node: "
                                    f"{len(response_tokens)} tokens "
                                    f"({len(request.prompt_token_ids) - prefix_boundary} user "
                                    f"+ {len(request.output_token_ids)} output) "
                                    f"for {request_id[:12]}"
                                )

                            if path:
                                self.turn_cache.release(path)
                        except Exception as e:
                            logger.debug(f"[turn_cache] store failed for {request_id}: {e}")
```

with:

```python
                elif self.turn_cache is not None:
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            segments = self._messages_to_segments(request)
                            path = getattr(request, "_turn_cache_path", [])
                            matched_depth = len(path)
                            parent = path[-1] if path else self.turn_cache.root
                            new_segments = segments[matched_depth:]
                            _turn_boundaries = getattr(request, "_turn_boundaries", None) or []
                            _boundary_states = getattr(request, "_boundary_states", None) or {}
                            parent_before_user = parent

                            for i, segment in enumerate(new_segments):
                                abs_idx = matched_depth + i
                                is_sys = segment.role == "system" and abs_idx == 0
                                is_last = i == len(new_segments) - 1

                                if is_last:
                                    state = None
                                elif abs_idx < len(_turn_boundaries):
                                    state = _boundary_states.get(_turn_boundaries[abs_idx])
                                else:
                                    state = None

                                parent = self.turn_cache.insert(
                                    parent, segment, [], [], state, is_system_prompt=is_sys
                                )
                                if not is_last:
                                    parent_before_user = parent

                            # Store completed exchange as a response node under parent_before_user.
                            last_boundary = _turn_boundaries[-1] if _turn_boundaries else 0
                            if (
                                request.output_token_ids
                                and last_boundary > 0
                                and request.prompt_token_ids
                                and new_segments
                            ):
                                from .turn_prefix_cache import Segment as _Seg
                                response_tokens = (
                                    list(request.prompt_token_ids[last_boundary:])
                                    + list(request.output_token_ids)
                                )
                                ec = request._extracted_cache
                                if isinstance(ec, list) and ec and isinstance(ec[0], dict):
                                    resp_state = ec
                                else:
                                    resp_state = self._extract_cache_states(ec) or None
                                self.turn_cache.insert(
                                    parent_before_user,
                                    _Seg(role="conversation", token_ids=response_tokens),
                                    [], [], resp_state, is_system_prompt=False,
                                )
                                logger.info(
                                    f"[turn_cache] stored response node: "
                                    f"{len(response_tokens)} tokens "
                                    f"({len(request.prompt_token_ids) - last_boundary} user "
                                    f"+ {len(request.output_token_ids)} output) "
                                    f"for {request_id[:12]}"
                                )

                            if path:
                                self.turn_cache.release(path)
                        except Exception as e:
                            logger.debug(f"[turn_cache] store failed for {request_id}: {e}")
```

Also in `_cleanup_finished`, find the section that evaluates boundary-captured tensors (around line 2724):

```python
            for _state_attr in ("_sys_prompt_state", "_conv_end_state"):
                _state = getattr(request, _state_attr, None) if request is not None else None
                if _state and isinstance(_state, list):
                    for layer_dict in _state:
                        if isinstance(layer_dict, dict) and "state" in layer_dict:
                            mx.eval(*layer_dict["state"])
```

Replace with:

```python
            _b_states = getattr(request, "_boundary_states", None) if request is not None else None
            if _b_states and isinstance(_b_states, dict):
                for _bstate in _b_states.values():
                    if isinstance(_bstate, list):
                        for layer_dict in _bstate:
                            if isinstance(layer_dict, dict) and "state" in layer_dict:
                                mx.eval(*layer_dict["state"])
```

- [ ] **Step 4: Run targeted tests**

```bash
python -m pytest tests/test_turn_prefix_cache.py -k "store_side or boundary_states" -xvs 2>&1 | tail -20
```

Expected: PASSED

- [ ] **Step 5: Run full test suite (excluding known-broken old tests)**

```bash
python -m pytest tests/test_turn_prefix_cache.py -x --ignore-glob="*prefix_boundary*" 2>&1 | tail -30
```

- [ ] **Step 6: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn_cache): _cleanup_finished store loop uses _boundary_states"
```

---

## Task 7: Boundary-aware chunk loop in `scheduler.py`

**Files:**
- Modify: `vllm_mlx/scheduler.py` — `_chunked_next` function and initial chunk setup (lines 509–660)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_turn_prefix_cache.py`:

```python
def test_chunked_prefill_boundary_aware_first_chunk():
    """When _turn_boundaries=[B_sys] and B_sys <= budget, first chunk lands exactly on B_sys."""
    from unittest.mock import MagicMock, patch
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(
        use_turn_cache=True, chunked_prefill_tokens=100
    )

    # Simulate a request with B_sys=40, budget=100, prompt=80 tokens
    req = MagicMock()
    req._turn_boundaries = [40]
    req.cached_tokens = 0
    req.prompt_token_ids = list(range(80))

    # The first chunk should be 40 (landing on B_sys)
    _turn_boundaries = req._turn_boundaries
    cached = req.cached_tokens
    budget = 100

    boundaries_to_hit = sorted(b for b in _turn_boundaries if b > cached)
    assert boundaries_to_hit
    first_b = boundaries_to_hit[0]
    dist = first_b - cached
    first_chunk = min(dist, budget) if dist <= budget else budget
    assert first_chunk == 40, f"Expected first_chunk=40, got {first_chunk}"


def test_chunked_prefill_boundary_aware_skips_when_boundary_beyond_budget():
    """When B_sys > budget, full budget is used (boundary handled in next iteration)."""
    req_boundaries = [300]  # B_sys=300
    cached = 0
    budget = 100

    boundaries_to_hit = sorted(b for b in req_boundaries if b > cached)
    first_b = boundaries_to_hit[0]
    dist = first_b - cached
    first_chunk = min(dist, budget) if dist <= budget else budget
    assert first_chunk == budget, f"Expected first_chunk={budget}, got {first_chunk}"


def test_chunked_prefill_boundary_aware_continuation_lands_on_boundary():
    """In the continuation loop, chunk size lands exactly on next boundary."""
    _turn_boundaries = [40, 80]
    cached = 0
    processed_so_far = 40  # first chunk already landed on B_sys=40
    budget = 100

    total_pos = cached + processed_so_far
    next_b = next((b for b in sorted(_turn_boundaries) if b > total_pos), None)
    assert next_b == 80
    dist = next_b - total_pos
    n_to_process = min(dist, budget) if dist <= budget else budget
    assert n_to_process == 40
```

- [ ] **Step 2: Run to verify the logic tests pass**

```bash
python -m pytest tests/test_turn_prefix_cache.py -k "chunked_prefill_boundary" -xvs 2>&1 | tail -15
```

These are pure logic tests (no scheduler wiring), so they should PASS already. If not, fix the test assertions.

- [ ] **Step 3: Update `_needs_boundary_split` detection in `scheduler.py`**

In `_chunked_next` (around line 514–523), replace:

```python
                if requests is not None and uid_to_request_id is not None:
                    for _uid, _toks, *_ in batch_prompts:
                        _rid = uid_to_request_id.get(_uid)
                        _req = requests.get(_rid) if _rid else None
                        if _req and getattr(_req, "prefix_boundary", 0) > 0:
                            _needs_boundary_split = True
                            break
```

with:

```python
                if requests is not None and uid_to_request_id is not None:
                    for _uid, _toks, *_ in batch_prompts:
                        _rid = uid_to_request_id.get(_uid)
                        _req = requests.get(_rid) if _rid else None
                        if _req and getattr(_req, "_turn_boundaries", []):
                            _needs_boundary_split = True
                            break
```

- [ ] **Step 4: Update initial chunk sizing in `scheduler.py`**

In `_chunked_next` (around lines 593–605), replace:

```python
                    _first_chunk = budget
                    if _needs_boundary_split and len(batch_prompts) == 1:
                        _uid0 = uids[0]
                        _rid0 = uid_to_request_id.get(_uid0)
                        _req0 = requests.get(_rid0) if _rid0 else None
                        _pb = getattr(_req0, "prefix_boundary", 0) if _req0 else 0
                        _cached = getattr(_req0, "cached_tokens", 0) if _req0 else 0
                        _adjusted_pb = _pb - _cached
                        if 0 < _adjusted_pb < padded.shape[1] - prompt_checkpoint + 1:
                            # Use remainder so budget-sized loop iterations land exactly on
                            # the boundary (prevents bypassing budget for large boundaries).
                            _rem = _adjusted_pb % budget
                            _first_chunk = _rem if _rem > 0 else budget
```

with:

```python
                    _first_chunk = budget
                    if _needs_boundary_split and len(batch_prompts) == 1:
                        _uid0 = uids[0]
                        _rid0 = uid_to_request_id.get(_uid0)
                        _req0 = requests.get(_rid0) if _rid0 else None
                        _turn_bds = getattr(_req0, "_turn_boundaries", []) if _req0 else []
                        _cached = getattr(_req0, "cached_tokens", 0) if _req0 else 0
                        _bds_to_hit = sorted(b for b in _turn_bds if b > _cached)
                        if _bds_to_hit:
                            _dist = _bds_to_hit[0] - _cached
                            if 0 < _dist < padded.shape[1] - prompt_checkpoint + 1:
                                _first_chunk = min(_dist, budget)
```

- [ ] **Step 5: Update continuation chunk sizing in `scheduler.py`**

In `_chunked_next`, the continuation block (around lines 383–387), replace:

```python
            n_to_process = (
                min(budget, remaining - prompt_checkpoint)
                if remaining > prompt_checkpoint
                else 0
            )
```

with:

```python
            n_to_process = (
                min(budget, remaining - prompt_checkpoint)
                if remaining > prompt_checkpoint
                else 0
            )
            # Boundary-aware: shrink chunk to land exactly on the next boundary
            if n_to_process > 0 and len(partial.get("uids", [])) == 1:
                _uid0 = partial["uids"][0]
                _rid0 = (uid_to_request_id or {}).get(_uid0)
                _req0 = (requests or {}).get(_rid0) if _rid0 else None
                if _req0 is not None:
                    _turn_bds = getattr(_req0, "_turn_boundaries", [])
                    _cached0 = getattr(_req0, "cached_tokens", 0)
                    _total_pos = _cached0 + partial["processed"]
                    _next_b = next((b for b in sorted(_turn_bds) if b > _total_pos), None)
                    if _next_b is not None:
                        _dist = _next_b - _total_pos
                        if _dist <= budget:
                            n_to_process = min(_dist, remaining - prompt_checkpoint)
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_turn_prefix_cache.py -k "chunked_prefill" -xvs 2>&1 | tail -15
```

Expected: PASSED

- [ ] **Step 7: Commit**

```bash
git add vllm_mlx/scheduler.py tests/test_turn_prefix_cache.py
git commit -m "feat(turn_cache): boundary-aware chunk loop replaces % budget alignment trick"
```

---

## Task 8: Update and clean up obsolete tests

**Files:**
- Modify: `tests/test_turn_prefix_cache.py`

The tests below use `prefix_boundary`, `sys_end_boundary`, `turn_boundaries` (old fields), `_sys_prompt_state`, `_conv_end_state`, `_turn_boundary_states`, `_mid_prefill_last_save`. These need to be updated to use the new field surface.

- [ ] **Step 1: Update `test_messages_to_segments_uses_prefix_boundary`**

Replace this test entirely with a call to the new helpers added in Task 4:

```python
def test_messages_to_segments_uses_turn_boundaries():
    """Scheduler _messages_to_segments splits using _turn_boundaries."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    sys_tokens = list(range(10))
    user_tokens_hi = [100, 101]
    user_tokens_yo = [200, 201]

    B_sys = len(sys_tokens)

    req1 = _make_request_with_boundaries(sys_tokens + user_tokens_hi, [B_sys])
    req2 = _make_request_with_boundaries(sys_tokens + user_tokens_yo, [B_sys])

    segs1 = sched._messages_to_segments(req1)
    segs2 = sched._messages_to_segments(req2)

    assert len(segs1) == 2
    assert len(segs2) == 2
    assert segs1[0].token_ids == segs2[0].token_ids == sys_tokens
    assert segs1[0].role == "system"
    assert segs1[1].token_ids == user_tokens_hi
    assert segs2[1].token_ids == user_tokens_yo
```

- [ ] **Step 2: Update `test_cross_session_hit_via_prefix_boundary`**

Replace with:

```python
def test_cross_session_hit_via_turn_boundaries():
    """Session 1 stores sys-prompt segment; session 2 gets a hit on it."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(20))
    user_hi = [100]
    user_yo = [200]
    B_sys = len(sys_tokens)

    req1 = _make_request_with_boundaries(sys_tokens + user_hi, [B_sys])
    segs1 = sched._messages_to_segments(req1)
    assert len(segs1) == 2
    n_sys = cache.insert(cache.root, segs1[0], [], [], None, is_system_prompt=True)
    cache.insert(n_sys, segs1[1], [], [], None)

    req2 = _make_request_with_boundaries(sys_tokens + user_yo, [B_sys])
    segs2 = sched._messages_to_segments(req2)
    path, _ = cache.match(segs2)

    assert len(path) >= 1
    assert path[0] is n_sys
    cache.release(path)
```

- [ ] **Step 3: Update `test_segment1_tokens_stable_across_turns`**

Replace with:

```python
def test_segment1_tokens_stable_across_turns_new():
    """Segment 1 (sys) has identical token_ids for turn 1, 2, and 3 using new boundaries."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    sys_tokens = list(range(100))
    u1 = list(range(100, 120)); a1 = list(range(200, 250))
    u2 = list(range(300, 315)); a2 = list(range(400, 430))
    u3 = list(range(500, 510))
    B_sys = len(sys_tokens)

    req1 = _make_request_with_boundaries(sys_tokens + u1, [B_sys])
    B_1 = B_sys + len(u1) + len(a1)
    req2 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2, [B_sys, B_1])
    B_2 = B_1 + len(u2) + len(a2)
    req3 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2 + a2 + u3, [B_sys, B_1, B_2])

    segs1 = sched._messages_to_segments(req1)
    segs2 = sched._messages_to_segments(req2)
    segs3 = sched._messages_to_segments(req3)

    assert segs1[0].token_ids == sys_tokens
    assert segs2[0].token_ids == sys_tokens
    assert segs3[0].token_ids == sys_tokens

    from vllm_mlx.turn_prefix_cache import _context_hash, TurnPrefixCache, TurnPrefixCacheConfig
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    h1 = _context_hash(cache.root.context_hash, segs1[0].token_ids)
    h2 = _context_hash(cache.root.context_hash, segs2[0].token_ids)
    h3 = _context_hash(cache.root.context_hash, segs3[0].token_ids)
    assert h1 == h2 == h3
```

- [ ] **Step 4: Update `test_multi_turn_sys_hit_each_turn`**

Replace with:

```python
def test_multi_turn_sys_hit_each_turn_new():
    """After storing turn 1, turns 2 and 3 must get a trie HIT on the sys segment."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(50))
    u1 = list(range(50, 60)); a1 = list(range(100, 120))
    u2 = list(range(200, 210)); a2 = list(range(300, 315))
    u3 = list(range(400, 405))
    B_sys = len(sys_tokens)

    # Turn 1 store
    req1 = _make_request_with_boundaries(sys_tokens + u1, [B_sys])
    segs1 = sched._messages_to_segments(req1)
    assert len(segs1) == 2
    sys_state = _make_extracted_state(n_layers=1, n_tokens=B_sys)
    sys_node = cache.insert(cache.root, segs1[0], [], [], sys_state, is_system_prompt=True)
    cache.insert(sys_node, segs1[1], [], [], None)

    # Turn 2 fetch: sys hit
    B_1 = B_sys + len(u1) + len(a1)
    req2 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2, [B_sys, B_1])
    segs2 = sched._messages_to_segments(req2)
    assert len(segs2) == 3
    path2, _ = cache.match(segs2)
    assert len(path2) >= 1
    assert path2[0] is sys_node
    assert sum(len(n.token_ids) for n in path2) == B_sys
    cache.release(path2)

    # Turn 3 fetch: sys hit
    B_2 = B_1 + len(u2) + len(a2)
    req3 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2 + a2 + u3, [B_sys, B_1, B_2])
    segs3 = sched._messages_to_segments(req3)
    assert len(segs3) == 4
    path3, _ = cache.match(segs3)
    assert len(path3) >= 1
    assert path3[0] is sys_node
    cache.release(path3)
```

- [ ] **Step 5: Update `test_conv_segment_grows_each_turn`**

Replace with:

```python
def test_conv_segment_grows_each_turn_new():
    """Conv segment at turn 2 differs from conv segment at turn 3."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)

    sys_tokens = list(range(20))
    u1 = list(range(20, 25)); a1 = list(range(100, 105))
    u2 = list(range(200, 203)); a2 = list(range(300, 306))
    u3 = list(range(400, 402))
    B_sys = len(sys_tokens)
    B_1 = B_sys + len(u1) + len(a1)
    B_2 = B_1 + len(u2) + len(a2)

    req2 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2, [B_sys, B_1])
    req3 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2 + a2 + u3, [B_sys, B_1, B_2])

    segs2 = sched._messages_to_segments(req2)
    segs3 = sched._messages_to_segments(req3)

    assert segs2[1].role == "conversation"
    assert segs3[1].role == "conversation"
    assert segs2[1].token_ids != segs3[1].token_ids
```

- [ ] **Step 6: Update `test_multi_turn_conv_stored_after_turn2`**

Replace with:

```python
def test_multi_turn_conv_stored_after_turn2_new():
    """After turn 2 completes, the trie contains a conv node under sys_node."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(20))
    u1 = list(range(20, 25)); a1 = list(range(100, 105))
    u2 = list(range(200, 203))
    B_sys = len(sys_tokens)
    B_1 = B_sys + len(u1) + len(a1)

    # Turn 1 store
    sys_state = _make_extracted_state(n_layers=1, n_tokens=B_sys)
    req1 = _make_request_with_boundaries(sys_tokens + u1, [B_sys])
    segs1 = sched._messages_to_segments(req1)
    sys_node = cache.insert(cache.root, segs1[0], [], [], sys_state, is_system_prompt=True)
    cache.insert(sys_node, segs1[1], [], [], None)

    # Turn 2: HIT on sys, then store [conv, user2]
    conv_state = _make_extracted_state(n_layers=1, n_tokens=B_1)
    req2 = MagicMock()
    req2.prompt_token_ids = sys_tokens + u1 + a1 + u2
    req2._turn_boundaries = [B_sys, B_1]
    req2._boundary_states = {B_sys: sys_state, B_1: conv_state}
    req2._turn_cache_path = [sys_node]
    req2._extracted_cache = _make_extracted_state(n_layers=1, n_tokens=B_1 + len(u2))
    req2.output_token_ids = [999]
    sys_node.ref_count += 1

    segs2 = sched._messages_to_segments(req2)
    assert len(segs2) == 3

    matched_depth = 1
    path = [sys_node]
    parent = sys_node
    new_segments = segs2[matched_depth:]
    _turn_boundaries = req2._turn_boundaries
    _boundary_states = req2._boundary_states
    parent_before_user = parent

    inserted = []
    for i, segment in enumerate(new_segments):
        abs_idx = matched_depth + i
        is_sys = segment.role == "system" and abs_idx == 0
        is_last = i == len(new_segments) - 1
        state = None if is_last else _boundary_states.get(_turn_boundaries[abs_idx]) if abs_idx < len(_turn_boundaries) else None
        parent = cache.insert(parent, segment, [], [], state, is_system_prompt=is_sys)
        if not is_last:
            parent_before_user = parent
        inserted.append(parent)

    cache.release(path)
    conv_node = inserted[0]
    user2_node = inserted[1]

    assert conv_node in sys_node.children.values()
    assert conv_node.recurrent_state is conv_state
    expected_conv_tokens = (sys_tokens + u1 + a1 + u2)[B_sys:B_1]
    assert conv_node.token_ids == expected_conv_tokens

    # Cross-session turn-2: 3-deep HIT
    req_cross2 = _make_request_with_boundaries(sys_tokens + u1 + a1 + u2, [B_sys, B_1])
    segs_cross2 = sched._messages_to_segments(req_cross2)
    path_cross2, _ = cache.match(segs_cross2)
    assert len(path_cross2) == 3
    assert path_cross2[1] is conv_node
    cache.release(path_cross2)
```

- [ ] **Step 7: Update mid-prefill tests referencing old attributes**

Delete or replace:
- `test_mid_prefill_stores_sys_prompt_state_at_boundary` → replaced by `test_mid_prefill_saves_boundary_state` (Task 5)
- `test_mid_prefill_does_not_store_state_away_from_boundary` → replaced by `test_mid_prefill_does_not_save_away_from_boundary` (Task 5)
- `test_mid_prefill_does_not_store_state_away_from_boundary_past_interval` → delete (interval throttle no longer applies to turn_cache path)
- `test_store_side_sets_recurrent_state_on_system_segment` → replaced by `test_store_side_uses_boundary_states` (Task 6)
- `test_cross_session_system_prompt_cache_hit_with_real_state` → update to use `_turn_boundaries`

For `test_cross_session_system_prompt_cache_hit_with_real_state`, replace:

```python
def test_cross_session_system_prompt_cache_hit_with_real_state():
    """Full path: session 1 stores sys state; session 2 fetches it and skips sys prefill."""
    from unittest.mock import MagicMock
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig
    from vllm_mlx.turn_prefix_cache import TurnPrefixCache, TurnPrefixCacheConfig

    sched = object.__new__(Scheduler)
    sched.config = SchedulerConfig(use_turn_cache=True, chunked_prefill_tokens=8192)
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    cache = TurnPrefixCache(TurnPrefixCacheConfig(checkpoint_stride=0))
    sched.turn_cache = cache

    sys_tokens = list(range(20))
    user_hi = [100, 101]
    user_yo = [200, 201]
    B_sys = len(sys_tokens)

    sys_state = _make_extracted_state(n_layers=2, n_tokens=B_sys)

    # Session 1 store
    req1 = _make_request_with_boundaries(sys_tokens + user_hi, [B_sys])
    req1 = MagicMock()  # need full MagicMock for output_token_ids etc.
    req1.prompt_token_ids = sys_tokens + user_hi
    req1._turn_boundaries = [B_sys]
    req1._boundary_states = {B_sys: sys_state}
    req1._turn_cache_path = []

    segs1 = sched._messages_to_segments(req1)
    parent = cache.root
    for i, segment in enumerate(segs1):
        is_sys = segment.role == "system" and i == 0
        state = sys_state if is_sys else None
        parent = cache.insert(parent, segment, [], [], state, is_system_prompt=is_sys)

    # Session 2 fetch
    req2 = MagicMock()
    req2.prompt_token_ids = sys_tokens + user_yo
    req2._turn_boundaries = [B_sys]
    req2.request_id = "session2"

    segs2 = sched._messages_to_segments(req2)
    path, has_recurrent = cache.match(segs2)

    assert path, "Expected HIT on system segment"
    ancestor = cache.find_checkpoint_ancestor(path)
    assert ancestor is not None
    raw_state = ancestor.recurrent_state
    assert isinstance(raw_state, list) and isinstance(raw_state[0], dict)

    reconstructed = sched._reconstruct_cache_from_states(raw_state)
    assert reconstructed is not None

    req2.prompt_cache = reconstructed
    req2.cached_tokens = sum(len(n.token_ids) for n in path)
    req2.remaining_tokens = req2.prompt_token_ids[req2.cached_tokens:]

    assert req2.cached_tokens == B_sys
    assert req2.remaining_tokens == user_yo
    cache.release(path)
```

- [ ] **Step 8: Update `_make_multi_turn_request`**

Replace `_make_multi_turn_request` with `_make_request_with_boundaries` (already added in Task 4 tests). Delete `_make_multi_turn_request`.

- [ ] **Step 9: Run the full test suite**

```bash
python -m pytest tests/test_turn_prefix_cache.py -x 2>&1 | tail -40
```

Expected: all tests PASS. If any old test is still failing (not yet updated), track it down and apply the same `_turn_boundaries` / `_boundary_states` translation.

- [ ] **Step 10: Commit**

```bash
git add tests/test_turn_prefix_cache.py
git commit -m "test(turn_cache): update all tests to use _turn_boundaries / _boundary_states"
```

---

## Task 9: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -x --timeout=120 2>&1 | tail -40
```

Expected: all tests PASS. Any failure here is a regression; investigate before marking complete.

- [ ] **Step 2: Verify no old field names remain in source**

```bash
grep -rn "prefix_boundary\|sys_end_boundary\|_sys_prompt_state\|_conv_end_state\|_turn_boundary_states\|_mid_prefill_last_save" \
  vllm_mlx/request.py vllm_mlx/engine_core.py vllm_mlx/engine/batched.py
```

Expected: no output. If any remain, they are stranded references — remove them.

- [ ] **Step 3: Verify `turn_boundaries` (old meaning) is gone from scheduler**

```bash
grep -n "turn_boundaries\b" vllm_mlx/scheduler.py | grep -v "_turn_boundaries\|# "
```

Expected: no output except where the word appears as a local variable name within the `_make_mid_prefill_save_callback` memory_aware_cache branch (that branch still reads old fields for backwards compatibility with non-turn-cache paths — that is intentional).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(turn_cache): final cleanup — verify no stale boundary fields remain"
```

---

## Spec Coverage Check

| Spec Section | Task |
|---|---|
| 1. Boundary Computation — `_compute_turn_boundaries` | Task 1 |
| 2. Prefill Alignment — boundary-aware chunk loop | Task 7 |
| 3. Trie Structure — `_messages_to_segments` + cleanup | Tasks 4, 6 |
| 4. Request Lifecycle — scheduling lookup (fetch) uses `_turn_boundaries` | Tasks 4, 2 |
| 4. Request Lifecycle — prefill captures `_boundary_states` | Task 5 |
| 4. Request Lifecycle — cleanup store loop | Task 6 |
| 4. Request fields delta — removed 7, added 2 | Tasks 2, 3 |
| Tests — boundary computation, tokenizer-only | Tasks 1, 4, 5, 6, 8 |

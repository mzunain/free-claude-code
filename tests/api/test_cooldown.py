"""Tests for api.cooldown.CooldownStore."""

from __future__ import annotations

from api.cooldown import (
    DEFAULT_COOLDOWN_SECONDS,
    CooldownStore,
    candidate_key,
)


def test_default_seconds_property_reflects_constructor():
    store = CooldownStore(default_seconds=7.5)
    assert store.default_seconds == 7.5


def test_default_seconds_uses_module_default_when_omitted():
    store = CooldownStore()
    assert store.default_seconds == DEFAULT_COOLDOWN_SECONDS


def test_mark_then_is_cooled_with_explicit_clock():
    store = CooldownStore(default_seconds=10.0)
    store.mark("nvidia_nim/qwen", now=100.0)
    assert store.is_cooled("nvidia_nim/qwen", now=100.0) is True
    assert store.is_cooled("nvidia_nim/qwen", now=109.999) is True
    assert store.is_cooled("nvidia_nim/qwen", now=110.0) is False


def test_is_cooled_evicts_expired_entries():
    store = CooldownStore(default_seconds=10.0)
    store.mark("p/m", now=0.0)
    assert "p/m" in store._until  # noqa: SLF001 - assert internal eviction
    assert store.is_cooled("p/m", now=999.0) is False
    assert "p/m" not in store._until  # noqa: SLF001 - assert internal eviction


def test_mark_never_shortens_existing_cooldown():
    store = CooldownStore(default_seconds=30.0)
    # First mark for 30s starting at t=0 -> expires t=30
    store.mark("p/m", now=0.0)
    # Second mark with shorter window 1s -> would expire t=2; existing wins
    store.mark("p/m", seconds=1.0, now=1.0)
    assert store.is_cooled("p/m", now=15.0) is True
    assert store.is_cooled("p/m", now=29.9) is True
    assert store.is_cooled("p/m", now=30.1) is False


def test_mark_extends_when_new_expiry_is_later():
    store = CooldownStore(default_seconds=10.0)
    store.mark("p/m", now=0.0)  # expires at 10
    store.mark("p/m", seconds=60.0, now=5.0)  # extends to 65
    assert store.is_cooled("p/m", now=64.0) is True
    assert store.is_cooled("p/m", now=66.0) is False


def test_mark_with_zero_default_is_noop():
    store = CooldownStore(default_seconds=0.0)
    expiry = store.mark("p/m", now=100.0)
    assert expiry == 0.0
    assert store.is_cooled("p/m", now=100.0) is False


def test_remaining_seconds():
    store = CooldownStore(default_seconds=20.0)
    store.mark("p/m", now=0.0)
    assert store.remaining("p/m", now=0.0) == 20.0
    assert store.remaining("p/m", now=15.0) == 5.0
    assert store.remaining("p/m", now=20.0) == 0.0
    assert store.remaining("missing", now=0.0) == 0.0


def test_partition_preserves_order_within_each_group():
    store = CooldownStore(default_seconds=10.0)
    items = ["a", "b", "c", "d", "e"]
    store.mark("b", now=0.0)
    store.mark("d", now=0.0)
    healthy, cooled = store.partition(items, key=lambda x: x, now=0.0)
    assert healthy == ["a", "c", "e"]
    assert cooled == ["b", "d"]


def test_reorder_puts_healthy_first_then_cooled():
    store = CooldownStore(default_seconds=10.0)
    items = ["a", "b", "c", "d"]
    store.mark("a", now=0.0)
    store.mark("c", now=0.0)
    result = store.reorder(items, key=lambda x: x, now=0.0)
    assert result == ["b", "d", "a", "c"]


def test_reorder_with_no_cooldowns_preserves_order():
    store = CooldownStore(default_seconds=10.0)
    items = ["x", "y", "z"]
    assert store.reorder(items, key=lambda x: x, now=0.0) == items


def test_reorder_with_all_cooldowns_returns_original_order():
    """Graceful degradation: if everything is cooled, don't starve the request."""
    store = CooldownStore(default_seconds=10.0)
    items = ["a", "b", "c"]
    for k in items:
        store.mark(k, now=0.0)
    assert store.reorder(items, key=lambda x: x, now=0.0) == ["a", "b", "c"]


def test_reset_clears_all_state():
    store = CooldownStore(default_seconds=10.0)
    store.mark("a", now=0.0)
    store.mark("b", now=0.0)
    store.reset()
    assert store.is_cooled("a", now=0.0) is False
    assert store.is_cooled("b", now=0.0) is False


def test_candidate_key_format():
    assert (
        candidate_key("nvidia_nim", "qwen/qwen3-coder-480b-a35b-instruct")
        == "nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct"
    )

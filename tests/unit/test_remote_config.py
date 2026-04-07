# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Unit tests for beagle_node.config.remote (BootstrapConfig, load_bootstrap,
RemoteConfigFetcher) and the _verify_secret_hash helper in beagle_server.api.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from beagle_node.config.remote import BootstrapConfig, RemoteConfigFetcher, load_bootstrap
from beagle_node.config.schema import NodeConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SERVER_URL = "http://tdoa.test"
_NODE_ID = "test-node-one"
_NODE_SECRET = "test-secret-xyz-1234"

# Minimal valid NodeConfig dict (freq_hop mode with all required fields)
_NODE_CONFIG_DICT = {
    "node_id": "test-node-one",
    "location": {
        "latitude_deg": 47.6,
        "longitude_deg": -122.3,
        "altitude_m": 50.0,
    },
    "sdr_mode": "freq_hop",
    "freq_hop": {},
    "sync_signal": {
        "primary_station": {
            "station_id": "KISW_99.9",
            "frequency_hz": 99_900_000.0,
            "latitude_deg": 47.625,
            "longitude_deg": -122.356,
        }
    },
    "target_channels": [{"frequency_hz": 155_100_000.0, "label": "TEST"}],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bootstrap(**kwargs) -> BootstrapConfig:
    defaults = {
        "server_url": _SERVER_URL,
        "node_id": _NODE_ID,
        "node_secret": _NODE_SECRET,
    }
    defaults.update(kwargs)
    return BootstrapConfig(**defaults)


def _make_fetcher(bootstrap: BootstrapConfig | None = None) -> RemoteConfigFetcher:
    return RemoteConfigFetcher(bootstrap or _make_bootstrap())


def _server_payload(version: int = 1) -> dict:
    """Simulates a successful GET /api/v1/nodes/{node_id}/config response."""
    return {
        "status": "ok",
        "node_id": _NODE_ID,
        "config_version": version,
        "config": _NODE_CONFIG_DICT,
    }


# ---------------------------------------------------------------------------
# BootstrapConfig
# ---------------------------------------------------------------------------

def test_bootstrap_config_accepts_required_fields() -> None:
    bs = _make_bootstrap()
    assert bs.server_url == _SERVER_URL
    assert bs.node_id == _NODE_ID
    assert bs.node_secret == _NODE_SECRET


def test_bootstrap_config_default_cache_path() -> None:
    assert _make_bootstrap().config_cache_path == "/var/cache/beagle/node_config.json"


def test_bootstrap_config_default_poll_interval() -> None:
    assert _make_bootstrap().config_poll_interval_s == 60.0


def test_bootstrap_config_default_register_on_start() -> None:
    assert _make_bootstrap().register_on_start is True


def test_bootstrap_config_custom_cache_path() -> None:
    bs = _make_bootstrap(config_cache_path="/tmp/my_cache.json")
    assert bs.config_cache_path == "/tmp/my_cache.json"


def test_bootstrap_config_missing_server_url_raises() -> None:
    with pytest.raises(Exception):
        BootstrapConfig(node_id=_NODE_ID, node_secret=_NODE_SECRET)


def test_bootstrap_config_missing_node_id_raises() -> None:
    with pytest.raises(Exception):
        BootstrapConfig(server_url=_SERVER_URL, node_secret=_NODE_SECRET)


def test_bootstrap_config_missing_node_secret_raises() -> None:
    with pytest.raises(Exception):
        BootstrapConfig(server_url=_SERVER_URL, node_id=_NODE_ID)


# ---------------------------------------------------------------------------
# load_bootstrap
# ---------------------------------------------------------------------------

def test_load_bootstrap_reads_yaml(tmp_path: Path) -> None:
    f = tmp_path / "bootstrap.yaml"
    f.write_text(
        "server_url: http://tdoa.example.com\n"
        "node_id: my-node-01\n"
        "node_secret: super-secret-xyz\n"
    )
    bs = load_bootstrap(str(f))
    assert bs.server_url == "http://tdoa.example.com"
    assert bs.node_id == "my-node-01"
    assert bs.node_secret == "super-secret-xyz"


def test_load_bootstrap_optional_fields(tmp_path: Path) -> None:
    f = tmp_path / "bootstrap.yaml"
    f.write_text(
        "server_url: http://tdoa.example.com\n"
        "node_id: my-node-01\n"
        "node_secret: secret\n"
        "config_poll_interval_s: 30\n"
        "register_on_start: false\n"
        "config_cache_path: /tmp/cache.json\n"
    )
    bs = load_bootstrap(str(f))
    assert bs.config_poll_interval_s == 30.0
    assert bs.register_on_start is False
    assert bs.config_cache_path == "/tmp/cache.json"


def test_load_bootstrap_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bootstrap(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# _verify_secret_hash  (module-level helper in beagle_server.api)
# ---------------------------------------------------------------------------

def test_verify_secret_sha256_correct() -> None:
    from beagle_server.api import _verify_secret_hash
    plaintext = "my-super-secret"
    stored = "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()
    assert _verify_secret_hash(plaintext, stored) is True


def test_verify_secret_sha256_wrong_plaintext() -> None:
    from beagle_server.api import _verify_secret_hash
    stored = "sha256:" + hashlib.sha256(b"correct-value").hexdigest()
    assert _verify_secret_hash("wrong-value", stored) is False


def test_verify_secret_unknown_scheme_returns_false() -> None:
    from beagle_server.api import _verify_secret_hash
    assert _verify_secret_hash("secret", "bcrypt:$2b$12$xxx") is False


def test_verify_secret_empty_stored_hash_returns_false() -> None:
    from beagle_server.api import _verify_secret_hash
    assert _verify_secret_hash("anything", "") is False


def test_verify_secret_plaintext_prefix_not_accepted() -> None:
    from beagle_server.api import _verify_secret_hash
    # Storing plaintext with no recognised prefix must not match
    assert _verify_secret_hash("secret", "plaintext:secret") is False


# ---------------------------------------------------------------------------
# RemoteConfigFetcher._fetch_from_server
# ---------------------------------------------------------------------------

def test_fetch_from_server_200_returns_node_config(httpx_mock) -> None:
    httpx_mock.add_response(json=_server_payload(version=3), status_code=200)
    fetcher = _make_fetcher()
    config = fetcher._fetch_from_server()
    assert isinstance(config, NodeConfig)
    assert config.node_id == _NODE_ID


def test_fetch_from_server_updates_current_version(httpx_mock) -> None:
    httpx_mock.add_response(json=_server_payload(version=7), status_code=200)
    fetcher = _make_fetcher()
    assert fetcher._current_version == 0
    fetcher._fetch_from_server()
    assert fetcher._current_version == 7


def test_fetch_from_server_304_returns_none(httpx_mock) -> None:
    httpx_mock.add_response(status_code=304)
    fetcher = _make_fetcher()
    assert fetcher._fetch_from_server() is None


def test_fetch_from_server_401_raises_runtime_error(httpx_mock) -> None:
    httpx_mock.add_response(status_code=401)
    fetcher = _make_fetcher()
    with pytest.raises(RuntimeError, match="authentication failed"):
        fetcher._fetch_from_server()


def test_fetch_from_server_403_raises_runtime_error(httpx_mock) -> None:
    httpx_mock.add_response(status_code=403)
    fetcher = _make_fetcher()
    with pytest.raises(RuntimeError, match="not found"):
        fetcher._fetch_from_server()


def test_fetch_from_server_pending_raises_runtime_error(httpx_mock) -> None:
    httpx_mock.add_response(
        json={"status": "pending", "config_version": 0, "config": None},
        status_code=200,
    )
    fetcher = _make_fetcher()
    with pytest.raises(RuntimeError, match="no config assigned"):
        fetcher._fetch_from_server()


def test_fetch_from_server_500_returns_none(httpx_mock) -> None:
    httpx_mock.add_response(status_code=500)
    assert _make_fetcher()._fetch_from_server() is None


def test_fetch_from_server_transport_error_returns_none(httpx_mock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    assert _make_fetcher()._fetch_from_server() is None


def test_fetch_from_server_sends_bearer_token(httpx_mock) -> None:
    httpx_mock.add_response(json=_server_payload(), status_code=200)
    _make_fetcher()._fetch_from_server()
    req = httpx_mock.get_requests()[0]
    assert req.headers["Authorization"] == f"Bearer {_NODE_SECRET}"


def test_fetch_from_server_sends_node_id_header(httpx_mock) -> None:
    httpx_mock.add_response(json=_server_payload(), status_code=200)
    _make_fetcher()._fetch_from_server()
    req = httpx_mock.get_requests()[0]
    assert req.headers["X-Node-ID"] == _NODE_ID


def test_fetch_from_server_calls_correct_url(httpx_mock) -> None:
    httpx_mock.add_response(json=_server_payload(), status_code=200)
    _make_fetcher()._fetch_from_server()
    req = httpx_mock.get_requests()[0]
    assert str(req.url) == f"{_SERVER_URL}/api/v1/nodes/{_NODE_ID}/config"


def test_fetch_from_server_strips_trailing_slash_from_base(httpx_mock) -> None:
    httpx_mock.add_response(json=_server_payload(), status_code=200)
    bs = _make_bootstrap(server_url=f"{_SERVER_URL}/")  # trailing slash
    RemoteConfigFetcher(bs)._fetch_from_server()
    req = httpx_mock.get_requests()[0]
    assert str(req.url) == f"{_SERVER_URL}/api/v1/nodes/{_NODE_ID}/config"


# ---------------------------------------------------------------------------
# RemoteConfigFetcher cache: _save_cache / _load_cache
# ---------------------------------------------------------------------------

def test_save_and_load_cache_round_trip(tmp_path: Path) -> None:
    cache_path = str(tmp_path / "config.json")
    bs = _make_bootstrap(config_cache_path=cache_path)
    fetcher = RemoteConfigFetcher(bs)
    original = NodeConfig.model_validate(_NODE_CONFIG_DICT)
    fetcher._save_cache(original)
    loaded = fetcher._load_cache()
    assert loaded is not None
    assert loaded.node_id == original.node_id
    assert loaded.target_channels[0].frequency_hz == original.target_channels[0].frequency_hz


def test_load_cache_returns_none_when_file_missing(tmp_path: Path) -> None:
    bs = _make_bootstrap(config_cache_path=str(tmp_path / "no_such.json"))
    assert RemoteConfigFetcher(bs)._load_cache() is None


def test_load_cache_returns_none_for_corrupt_json(tmp_path: Path) -> None:
    cache_file = tmp_path / "config.json"
    cache_file.write_text("{ this is not valid json {{ ")
    bs = _make_bootstrap(config_cache_path=str(cache_file))
    assert RemoteConfigFetcher(bs)._load_cache() is None


def test_load_cache_returns_none_for_invalid_schema(tmp_path: Path) -> None:
    """Valid JSON but not a valid NodeConfig."""
    cache_file = tmp_path / "config.json"
    cache_file.write_text(json.dumps({"not": "a node config"}))
    bs = _make_bootstrap(config_cache_path=str(cache_file))
    assert RemoteConfigFetcher(bs)._load_cache() is None


def test_save_cache_creates_missing_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "config.json"
    bs = _make_bootstrap(config_cache_path=str(nested))
    fetcher = RemoteConfigFetcher(bs)
    config = NodeConfig.model_validate(_NODE_CONFIG_DICT)
    fetcher._save_cache(config)  # must not raise
    assert nested.exists()


def test_save_cache_is_valid_json(tmp_path: Path) -> None:
    cache_path = tmp_path / "config.json"
    bs = _make_bootstrap(config_cache_path=str(cache_path))
    fetcher = RemoteConfigFetcher(bs)
    fetcher._save_cache(NodeConfig.model_validate(_NODE_CONFIG_DICT))
    # Must be parseable
    data = json.loads(cache_path.read_text())
    assert data["node_id"] == _NODE_ID


# ---------------------------------------------------------------------------
# RemoteConfigFetcher.fetch_initial_config
# ---------------------------------------------------------------------------

def test_fetch_initial_config_returns_config_on_success(httpx_mock, tmp_path: Path) -> None:
    httpx_mock.add_response(json=_server_payload(version=5), status_code=200)
    bs = _make_bootstrap(config_cache_path=str(tmp_path / "config.json"))
    config = RemoteConfigFetcher(bs).fetch_initial_config()
    assert isinstance(config, NodeConfig)
    assert config.node_id == _NODE_ID


def test_fetch_initial_config_writes_cache_on_success(httpx_mock, tmp_path: Path) -> None:
    httpx_mock.add_response(json=_server_payload(), status_code=200)
    cache_path = tmp_path / "config.json"
    bs = _make_bootstrap(config_cache_path=str(cache_path))
    RemoteConfigFetcher(bs).fetch_initial_config()
    assert cache_path.exists()


def test_fetch_initial_config_falls_back_to_cache(httpx_mock, tmp_path: Path) -> None:
    """Server unreachable -> falls back to a pre-existing cache file."""
    cache_path = str(tmp_path / "config.json")
    # Pre-populate the cache
    bs = _make_bootstrap(config_cache_path=cache_path)
    fetcher = RemoteConfigFetcher(bs)
    fetcher._save_cache(NodeConfig.model_validate(_NODE_CONFIG_DICT))

    # Now make the server unreachable
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    config = fetcher.fetch_initial_config()
    assert isinstance(config, NodeConfig)
    assert config.node_id == _NODE_ID


def test_fetch_initial_config_raises_when_no_server_no_cache(
    httpx_mock, tmp_path: Path
) -> None:
    """Neither server nor cache -> RuntimeError."""
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    bs = _make_bootstrap(config_cache_path=str(tmp_path / "config.json"))
    with pytest.raises(RuntimeError, match="Cannot reach server"):
        RemoteConfigFetcher(bs).fetch_initial_config()


def test_fetch_initial_config_raises_on_auth_error(httpx_mock, tmp_path: Path) -> None:
    """401 from server propagates as RuntimeError (not silently falls back to cache)."""
    httpx_mock.add_response(status_code=401)
    bs = _make_bootstrap(config_cache_path=str(tmp_path / "config.json"))
    with pytest.raises(RuntimeError, match="authentication failed"):
        RemoteConfigFetcher(bs).fetch_initial_config()


# ---------------------------------------------------------------------------
# RemoteConfigFetcher.set_heartbeat_data / _get_heartbeat_data
# ---------------------------------------------------------------------------

def test_set_heartbeat_data_stored_and_retrievable() -> None:
    fetcher = _make_fetcher()
    fetcher.set_heartbeat_data({"noise_floor_db": -55.0, "sdr_mode": "freq_hop"})
    hb = fetcher._get_heartbeat_data()
    assert hb["noise_floor_db"] == -55.0
    assert hb["sdr_mode"] == "freq_hop"


def test_set_heartbeat_data_returns_copy() -> None:
    """Mutations to the returned dict don't affect the stored data."""
    fetcher = _make_fetcher()
    fetcher.set_heartbeat_data({"noise_floor_db": -55.0})
    hb = fetcher._get_heartbeat_data()
    hb["noise_floor_db"] = 999
    assert fetcher._get_heartbeat_data()["noise_floor_db"] == -55.0


def test_set_heartbeat_data_overwrites_previous() -> None:
    fetcher = _make_fetcher()
    fetcher.set_heartbeat_data({"noise_floor_db": -55.0})
    fetcher.set_heartbeat_data({"noise_floor_db": -60.0, "clock_source": "gps_1pps"})
    hb = fetcher._get_heartbeat_data()
    assert hb["noise_floor_db"] == -60.0
    assert hb["clock_source"] == "gps_1pps"


# ---------------------------------------------------------------------------
# RemoteConfigFetcher._fetch_poll sends POST with heartbeat data
# ---------------------------------------------------------------------------

def test_fetch_poll_sends_post_with_heartbeat(httpx_mock) -> None:
    """_fetch_poll sends a POST (not GET) carrying heartbeat data."""
    httpx_mock.add_response(status_code=304)
    fetcher = _make_fetcher()
    fetcher.set_heartbeat_data({"noise_floor_db": -50.3, "sdr_mode": "rspduo"})
    fetcher._fetch_poll(wait_s=10)
    req = httpx_mock.get_requests()[0]
    assert req.method == "POST"
    body = json.loads(req.content)
    assert body["noise_floor_db"] == -50.3
    assert body["sdr_mode"] == "rspduo"


def test_fetch_poll_post_empty_heartbeat_when_none_set(httpx_mock) -> None:
    """Without set_heartbeat_data, _fetch_poll sends an empty body."""
    httpx_mock.add_response(status_code=304)
    fetcher = _make_fetcher()
    fetcher._fetch_poll(wait_s=10)
    req = httpx_mock.get_requests()[0]
    assert req.method == "POST"
    body = json.loads(req.content)
    assert body == {}


# ---------------------------------------------------------------------------
# RemoteConfigFetcher._fetch_poll: response code handling
# ---------------------------------------------------------------------------

def test_fetch_poll_304_returns_none(httpx_mock) -> None:
    """304 Not Modified is the normal long-poll timeout response."""
    httpx_mock.add_response(status_code=304)
    assert _make_fetcher()._fetch_poll(wait_s=10) is None


def test_fetch_poll_200_unchanged_version_returns_none(httpx_mock) -> None:
    """A 200 with the same config_version we already have is a no-op."""
    httpx_mock.add_response(json=_server_payload(version=0), status_code=200)
    fetcher = _make_fetcher()
    assert fetcher._current_version == 0
    assert fetcher._fetch_poll(wait_s=10) is None


def test_fetch_poll_500_raises_transient(httpx_mock) -> None:
    """5xx must raise so the poll loop applies exponential backoff."""
    from beagle_node.config.remote import _TransientPollError
    httpx_mock.add_response(status_code=500)
    with pytest.raises(_TransientPollError, match="HTTP 500"):
        _make_fetcher()._fetch_poll(wait_s=10)


def test_fetch_poll_503_raises_transient(httpx_mock) -> None:
    """503 Service Unavailable (e.g. uvicorn still starting) -> backoff."""
    from beagle_node.config.remote import _TransientPollError
    httpx_mock.add_response(status_code=503)
    with pytest.raises(_TransientPollError, match="HTTP 503"):
        _make_fetcher()._fetch_poll(wait_s=10)


def test_fetch_poll_4xx_raises_transient(httpx_mock) -> None:
    """4xx (e.g. wrong secret) is also non-retriable; back off rather
    than tight-loop while operator notices."""
    from beagle_node.config.remote import _TransientPollError
    httpx_mock.add_response(status_code=401)
    with pytest.raises(_TransientPollError, match="HTTP 401"):
        _make_fetcher()._fetch_poll(wait_s=10)


def test_fetch_poll_unparseable_json_raises_transient(httpx_mock) -> None:
    """A 200 with garbage body should not crash the poll loop."""
    from beagle_node.config.remote import _TransientPollError
    httpx_mock.add_response(status_code=200, content=b"not json")
    with pytest.raises(_TransientPollError, match="unparseable"):
        _make_fetcher()._fetch_poll(wait_s=10)


def test_fetch_poll_transport_error_propagates(httpx_mock) -> None:
    """TransportError still propagates (the loop already handles it)."""
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    with pytest.raises(httpx.ConnectError):
        _make_fetcher()._fetch_poll(wait_s=10)


# ---------------------------------------------------------------------------
# RemoteConfigFetcher._poll_loop: exponential backoff behaviour
# ---------------------------------------------------------------------------

def _drive_poll_loop(
    fetcher: RemoteConfigFetcher,
    poll_outcomes: list,
    max_iterations: int = 20,
):
    """Run _poll_loop in the foreground with mocked _fetch_poll, capturing
    the sequence of stop_event.wait() calls so the test can inspect the
    backoff schedule.

    poll_outcomes : list of values to return from successive _fetch_poll
        calls.  Each entry is either:
          - a NodeConfig (treated as "new config arrived")
          - None         (treated as "304 / no change")
          - an Exception instance (raised by _fetch_poll)
        The loop terminates after the list is exhausted (the test makes the
        next call set the stop event).
    """
    from beagle_node.config import remote as remote_module

    waits: list[float] = []
    iteration = {"n": 0}

    def _fake_fetch_poll(wait_s):
        i = iteration["n"]
        iteration["n"] += 1
        if i >= len(poll_outcomes):
            fetcher._stop_event.set()
            return None
        outcome = poll_outcomes[i]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def _fake_wait(timeout):
        waits.append(timeout)
        return fetcher._stop_event.is_set()

    # No-op random so backoff is deterministic
    real_uniform = remote_module.random.uniform
    remote_module.random.uniform = lambda a, b: (a + b) / 2

    fetcher._fetch_poll = _fake_fetch_poll  # type: ignore[assignment]
    fetcher._stop_event.wait = _fake_wait    # type: ignore[assignment]

    try:
        # Cap the loop in case of a bug -- forcibly stop after max_iterations
        # by counting waits.
        original_wait = _fake_wait

        def _capped_wait(timeout):
            r = original_wait(timeout)
            if iteration["n"] >= max_iterations:
                fetcher._stop_event.set()
            return r

        fetcher._stop_event.wait = _capped_wait  # type: ignore[assignment]
        fetcher._poll_loop(on_update=lambda c: None)
    finally:
        remote_module.random.uniform = real_uniform

    return waits, iteration["n"]


def test_poll_loop_backs_off_on_transient_error() -> None:
    """A run of 5xx errors should produce a strictly increasing backoff
    sequence (1, 2, 4, 8, ...) up to the cap."""
    from beagle_node.config.remote import _TransientPollError

    fetcher = _make_fetcher()
    # 6 consecutive errors, then the 7th call exhausts the script and the
    # helper sets the stop event (so the loop exits after one extra trip).
    outcomes = [_TransientPollError("HTTP 503") for _ in range(6)]
    waits, n_calls = _drive_poll_loop(fetcher, outcomes)

    # First wait is the initial jitter (uniform 0..5 -> 2.5 with our patch).
    # The remaining waits are the backoff sleeps after each error.
    assert n_calls == 7
    backoff_waits = waits[1:]   # skip the initial jitter
    assert len(backoff_waits) >= 6
    # Strictly increasing for the first few entries (until the 120s cap).
    # Initial 1.0s, doubled each step: 1, 2, 4, 8, 16, 32 ...
    assert backoff_waits[0] == pytest.approx(1.0, abs=0.01)
    assert backoff_waits[1] == pytest.approx(2.0, abs=0.01)
    assert backoff_waits[2] == pytest.approx(4.0, abs=0.01)
    assert backoff_waits[3] == pytest.approx(8.0, abs=0.01)
    assert backoff_waits[4] == pytest.approx(16.0, abs=0.01)
    assert backoff_waits[5] == pytest.approx(32.0, abs=0.01)


def test_poll_loop_backoff_caps_at_max() -> None:
    """After enough errors the backoff plateau at _BACKOFF_MAX_S."""
    from beagle_node.config.remote import _TransientPollError

    fetcher = _make_fetcher()
    # 10 errors -> 1, 2, 4, 8, 16, 32, 64, 120, 120, 120
    outcomes = [_TransientPollError("HTTP 503") for _ in range(10)]
    waits, _ = _drive_poll_loop(fetcher, outcomes)
    backoff_waits = waits[1:]
    # The last few entries should all equal _BACKOFF_MAX_S (120s).
    assert backoff_waits[-1] == pytest.approx(120.0, abs=0.01)
    assert backoff_waits[-2] == pytest.approx(120.0, abs=0.01)


def test_poll_loop_resets_backoff_after_success() -> None:
    """A successful poll between errors should reset the backoff to the
    initial value, so the next failure starts at 1s again."""
    from beagle_node.config.remote import _TransientPollError

    fetcher = _make_fetcher()
    # error, error, error (1, 2, 4), success, error (back to 1)
    outcomes = [
        _TransientPollError("HTTP 503"),
        _TransientPollError("HTTP 503"),
        _TransientPollError("HTTP 503"),
        None,                              # 304 / no change -> resets backoff
        _TransientPollError("HTTP 503"),
    ]
    waits, _ = _drive_poll_loop(fetcher, outcomes)
    backoff_waits = waits[1:]
    # 1, 2, 4 (errors) ... no wait after the success (loop continues
    # immediately) ... 1 (next error, reset)
    assert backoff_waits[0] == pytest.approx(1.0, abs=0.01)
    assert backoff_waits[1] == pytest.approx(2.0, abs=0.01)
    assert backoff_waits[2] == pytest.approx(4.0, abs=0.01)
    assert backoff_waits[3] == pytest.approx(1.0, abs=0.01)


def test_poll_loop_backs_off_on_transport_error() -> None:
    """httpx.TransportError takes the same backoff path as _TransientPollError."""
    fetcher = _make_fetcher()
    outcomes = [
        httpx.ConnectError("connection refused"),
        httpx.ConnectError("connection refused"),
        httpx.ConnectError("connection refused"),
    ]
    waits, _ = _drive_poll_loop(fetcher, outcomes)
    backoff_waits = waits[1:]
    assert backoff_waits[0] == pytest.approx(1.0, abs=0.01)
    assert backoff_waits[1] == pytest.approx(2.0, abs=0.01)
    assert backoff_waits[2] == pytest.approx(4.0, abs=0.01)

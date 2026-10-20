# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import HAS_DEPS, create_test_manager, make_mock_engine, mock_ray_get


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


@pytest.fixture
def wiring(monkeypatch):
    from relax.agentic import rollout as agentic
    from relax.distributed.ray import rollout
    from relax.engine.lora_versions import runtime

    events = []
    engine = make_mock_engine()
    server = SimpleNamespace(engines=[engine], engine_groups=[])
    manager = create_test_manager(servers={"default": server})
    manager.args.use_agentic_rollout = True
    manager.args.use_fault_tolerance = False
    manager._lora_runtime = None
    manager._lora_configuring = False
    manager._stop_eviction_monitor = lambda **_kwargs: None
    manager._shutdown_all_engines = lambda: events.append("engines-close")
    host = SimpleNamespace(
        start=SimpleNamespace(remote=lambda *args: events.append("start")),
        publish=SimpleNamespace(remote=lambda path: events.append("publish")),
        close=SimpleNamespace(remote=lambda: events.append("sessions-and-host-close")),
    )
    factory = Mock()
    factory.options.return_value.remote.return_value = host
    monkeypatch.setattr(runtime, "LoRAVersionRuntime", factory)
    monkeypatch.setattr(rollout, "with_control_plane_affinity", lambda args, options: options)
    monkeypatch.setattr(rollout.ray, "get", mock_ray_get)
    monkeypatch.setattr(rollout.ray, "kill", lambda actor: events.append("host-kill"))
    monkeypatch.setattr(rollout, "stop_launched_routers", lambda: events.append("routers-close"))
    monkeypatch.setattr(agentic, "bind_agentic_lora_runtime", lambda handle: events.append("sessions-bind"))
    return manager, host, events, agentic, factory


def test_lora_configuration_publishes_before_session_injection_and_disposes_in_order(wiring):
    manager, host, events, _agentic, _factory = wiring
    assert manager.configure_lora_versions("artifact", [1]) is host
    assert events == ["start", "publish", "sessions-bind"]
    assert manager.get_lora_version_runtime() is host
    with pytest.raises(RuntimeError, match="fixed engine pool"):
        manager.create_scale_out_request(num_replicas=2)
    with pytest.raises(RuntimeError, match="fixed engine pool"):
        manager.create_scale_in_request(num_replicas=1)
    manager.dispose()
    assert events[3:] == ["sessions-and-host-close", "host-kill", "engines-close", "routers-close"]


def test_lora_partial_configuration_retains_owner_and_blocks_pool_changes(wiring, monkeypatch):
    manager, host, _events, agentic, _factory = wiring
    monkeypatch.setattr(agentic, "bind_agentic_lora_runtime", Mock(side_effect=RuntimeError("injection failed")))
    with pytest.raises(RuntimeError, match="injection failed"):
        manager.configure_lora_versions("artifact", [1])
    assert manager._lora_runtime is host
    assert not manager._lora_configuring
    assert manager._lora_configuration_failed
    with pytest.raises(RuntimeError, match="incomplete"):
        manager._begin_generation()
    with pytest.raises(RuntimeError, match="already configured"):
        manager.configure_lora_versions("artifact", [1])
    manager.dispose()


def test_lora_actor_creation_failure_clears_configuring_but_fails_closed(wiring):
    manager, _host, _events, _agentic, factory = wiring
    factory.options.side_effect = RuntimeError("actor creation failed")
    with pytest.raises(RuntimeError, match="actor creation failed"):
        manager.configure_lora_versions("artifact", [1])
    assert not manager._lora_configuring
    with pytest.raises(RuntimeError, match="fixed engine pool"):
        manager._require_unmanaged_engine_pool()


def test_scale_out_rechecks_lora_mode_after_slow_endpoint_validation(wiring):
    manager, _host, _events, _agentic, _factory = wiring

    def collect(_server):
        manager._lora_configuring = True
        return set()

    manager._collect_existing_engine_addrs = collect
    with pytest.raises(RuntimeError, match="fixed engine pool"):
        manager.create_scale_out_request(engine_urls=["http://new:8000"])
    assert not manager._scale_out_requests


def test_lora_configuration_rejects_generation_that_already_started(wiring):
    manager, _host, events, _agentic, _factory = wiring
    manager._begin_generation()
    with pytest.raises(RuntimeError, match="before generation"):
        manager.configure_lora_versions("artifact", [1])
    assert not events

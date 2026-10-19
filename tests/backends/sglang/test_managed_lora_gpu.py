# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in checks on two fresh, dedicated SGLang engines on this host.

Set RELAX_LORA_TEST_URLS (comma-separated URLs), RELAX_LORA_TEST_ADAPTER_DIR
(SGLang-compatible full export: adapter_config.json +
adapter_model.safetensors), RELAX_LORA_TEST_BASE_FINGERPRINT,
RELAX_LORA_TEST_INPUT_IDS (JSON token IDs), and RELAX_LORA_TEST_CONTEXT_LENGTH.
Engines need this patch, LoRA enabled, DP=PP=1, max_loras_per_batch >= 3, one
tokenizer worker, and no PD/speculation/HiCache L3. Uses real shared memory,
HTTP, worker checksums and generation. Does not launch or stop any service. The
final default adapter remains loaded for inspection on these test engines.
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest

from relax.backends.sglang.lora_client import SGLangLoRAClient
from relax.engine.lora_versions.identity import snapshot_adapter
from relax.engine.lora_versions.manager import SessionKey, VersionState
from relax.engine.lora_versions.publisher import LoRAPublisher


pytestmark = pytest.mark.skipif(
    not os.environ.get("RELAX_LORA_TEST_URLS"),
    reason="Requires explicit dedicated same-host GPU endpoints and an adapter export; never launches services.",
)


@pytest.mark.asyncio
async def test_two_engine_publication_rejection_and_reclamation():
    from safetensors.torch import load_file

    urls = os.environ["RELAX_LORA_TEST_URLS"].split(",")
    assert len(urls) == 2 and len(set(urls)) == 2
    adapter_dir = Path(os.environ["RELAX_LORA_TEST_ADAPTER_DIR"])
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    probe_ids = json.loads(os.environ["RELAX_LORA_TEST_INPUT_IDS"])
    base = os.environ["RELAX_LORA_TEST_BASE_FINGERPRINT"]
    context_length = int(os.environ["RELAX_LORA_TEST_CONTEXT_LENGTH"])
    old = snapshot_adapter(1, tensors, config, base_fingerprint=base)
    # Distinct bytes and digest, while keeping the adapter valid for this base model.
    changed = {name: value.clone() for name, value in tensors.items()}
    first = next(value for value in changed.values() if value.is_floating_point())
    first.reshape(-1)[0] += 0.01
    new = snapshot_adapter(2, changed, config, base_fingerprint=base)
    assert old.identity.digest != new.identity.digest
    async with AsyncExitStack() as stack:
        clients = {
            str(i): SGLangLoRAClient(await stack.enter_async_context(httpx.AsyncClient(base_url=url, timeout=120)))
            for i, url in enumerate(urls)
        }
        publisher = await LoRAPublisher.connect(clients, probe_ids)
        manager = publisher.manager
        await publisher.publish(old)
        assert manager.serving_engines == ("0", "1")
        manager.register_shard("test", "test-instance")
        session = SessionKey("test", "test-instance", 0)
        assert manager.bind(session) == old.identity
        # P3: actual tokenization rejection after adapter acquire must release it.
        for client in clients.values():
            response = await client.http.post(
                "/generate",
                json={
                    "input_ids": [probe_ids[0]] * (context_length + 1),
                    "lora_path": old.identity.lora_name,
                    "sampling_params": {"max_new_tokens": 1},
                },
            )
            assert response.status_code == 400, response.text[:300]
            inventory = await client.inventory()
            assert all(row["requests"] == 0 for row in inventory["adapters"])
        await publisher.publish(new)
        assert manager.default_version == 2 and manager.publication(1).state == VersionState.RETIRED
        # Old sessions keep the old adapter usable after the default switches.
        await asyncio.gather(*(client.probe(old.identity.lora_name, probe_ids) for client in clients.values()))
        manager.release(session)
        for _ in range(100):
            assert not await publisher.reconcile()
            if manager.publication(1).state == VersionState.RECLAIMED:
                break
            await asyncio.sleep(0.05)
        assert manager.publication(1).state == VersionState.RECLAIMED
        for client in clients.values():
            response = await client.http.post(
                "/generate",
                json={
                    "input_ids": probe_ids,
                    "lora_path": old.identity.lora_name,
                    "sampling_params": {"max_new_tokens": 1},
                },
            )
            assert response.status_code == 409, response.text[:300]
            await client.probe(new.identity.lora_name, probe_ids)
            inventory = await client.inventory()
            assert sum(row["lora_id"] is not None for row in inventory["adapters"]) == 1

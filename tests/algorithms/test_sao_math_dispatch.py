# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise candidate mathematical routes without publishing a training
recipe."""

import dataclasses
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.advantages import compute_advantages_and_returns  # noqa: E402
from relax.algorithms.policy import compute_policy_loss_for  # noqa: E402
from relax.algorithms.spec import ALGORITHM_SPECS, get_algorithm  # noqa: E402


@pytest.fixture
def math_routes(monkeypatch):
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_context_parallel_world_size=lambda: 1)
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    # Local routing only: public SAO training waits for optimizer/critic integration.
    monkeypatch.setitem(
        ALGORITHM_SPECS,
        "ppo",
        dataclasses.replace(get_algorithm("ppo"), advantage_fn="masked_gae", policy_loss_fn="dis"),
    )
    return core.mpu


def _args(**overrides):
    config = dict(
        advantage_estimator="ppo",
        gamma=1.0,
        lambd=0.95,
        critic_gae_lambda=1.0,
        gae_observation_mode="skip",
        policy_gae_lambda_mode="fixed",
        policy_gae_alpha=None,
        kl_coef=0.0,
        context_parallel_size=1,
        dynamic_context_parallel=False,
        dis_epsilon_low=0.3,
        dis_epsilon_high=5.0,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _inputs():
    return dict(
        rewards=[1.0],
        kl=[torch.arange(5, dtype=torch.float32)],
        values=[torch.tensor([0.2, 9.0, -7.0, 0.4, 20.0])],
        loss_masks=[torch.tensor([1.0, 0, 0, 1, 0])],
        response_lengths=[5],
        total_lengths=[15],
    )


def test_shared_dispatch_runs_masked_targets_and_dis_without_mutating_masks_or_kl(math_routes):
    args = _args()
    inputs = _inputs()
    kl_before = inputs["kl"][0].clone()
    advantages, returns = compute_advantages_and_returns(args, **inputs)
    torch.testing.assert_close(advantages[0], torch.tensor([0.77, 0, 0, 0.6, 0]))
    torch.testing.assert_close(returns[0], torch.tensor([1.0, 0, 0, 1, 0]))
    torch.testing.assert_close(inputs["kl"][0], kl_before)
    current = torch.tensor([-1.0, math.nan, math.inf, -1.0, math.nan], requires_grad=True)
    loss, dropped, metrics = compute_policy_loss_for(
        args,
        log_probs=current,
        ppo_kl=torch.tensor([0, math.nan, math.inf, -math.log(10), math.nan]),
        advantages=advantages[0],
        loss_mask=inputs["loss_masks"][0],
    )
    (loss.sum() / inputs["loss_masks"][0].sum()).backward()
    torch.testing.assert_close(current.grad, torch.tensor([-0.385, 0, 0, 0, 0]))
    torch.testing.assert_close(dropped, torch.tensor([0.0, 0, 0, 1, 0]))
    assert metrics == {}
    assert inputs["loss_masks"][0].tolist() == [1, 0, 0, 1, 0]


@pytest.mark.parametrize("overrides", [{"context_parallel_size": 2}, {"dynamic_context_parallel": True}])
def test_new_gae_modes_reject_configured_cp_even_under_ppo_name(math_routes, overrides):
    with pytest.raises(ValueError, match="static CP=1"):
        compute_advantages_and_returns(_args(**overrides), **_inputs())


def test_new_gae_modes_check_the_actual_context_parallel_group(math_routes):
    math_routes.get_context_parallel_world_size = lambda: 2
    with pytest.raises(ValueError, match="static CP=1"):
        compute_advantages_and_returns(_args(), **_inputs())


def test_new_gae_modes_reject_unimplemented_kl_shaping_and_partial_values(math_routes):
    with pytest.raises(ValueError, match="kl-coef 0"):
        compute_advantages_and_returns(_args(kl_coef=0.1), **_inputs())
    inputs = _inputs()
    inputs["response_lengths"] = [10]
    with pytest.raises(ValueError, match="complete response values"):
        compute_advantages_and_returns(_args(), **inputs)


def test_dis_dispatch_requires_original_generation_mask(math_routes):
    with pytest.raises(ValueError, match="original generation mask"):
        compute_policy_loss_for(_args(), log_probs=torch.zeros(2), ppo_kl=torch.zeros(2), advantages=torch.ones(2))

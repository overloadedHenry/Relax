# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent recurrence and gradient oracles for the SAO mathematical
core."""

import math

import pytest


torch = pytest.importorskip("torch")

from relax.utils.training.ppo_utils import (  # noqa: E402
    compute_dis_loss,
    get_masked_advantages_and_returns_batch,
    vanilla_gae,
)


def _targets(values, masks, rewards, **options):
    config = dict(gamma=1.0, policy_lambd=0.95, critic_lambd=1.0, observation_mode="skip")
    config.update(options)
    return get_masked_advantages_and_returns_batch(values, rewards, masks, **config)


def _reference(values, mask, reward, gamma, policy_lambd, critic_lambd, mode):
    """Scalar Python oracle, deliberately independent of the parallel scan."""
    actions = [index for index, active in enumerate(mask) if active]
    timeline = actions if mode == "skip" else list(range(actions[-1] + 1))
    advantages = [0.0] * len(values)
    returns = [0.0] * len(values)
    for lambd, output, add_value in ((policy_lambd, advantages, False), (critic_lambd, returns, True)):
        carry = 0.0
        next_value = 0.0
        for index in reversed(timeline):
            token_reward = reward if index == actions[-1] else 0.0
            delta = token_reward + gamma * next_value - values[index]
            carry = delta + gamma * lambd * carry
            if mask[index]:
                output[index] = carry + (values[index] if add_value else 0.0)
            next_value = values[index]
    return advantages, returns


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_dis_closed_bounds_safe_exponent_and_detached_weights(sign):
    lower, upper = math.log(0.7), math.log(6.0)
    ratios = torch.tensor(
        [math.nextafter(lower, -math.inf), lower, 0.0, upper, math.nextafter(upper, math.inf), -1000, 1000],
        dtype=torch.float64,
    )
    current = torch.full_like(ratios, -2.0, requires_grad=True)
    kl = (-ratios).requires_grad_()
    advantages = torch.full_like(ratios, sign * 2, requires_grad=True)
    mask = torch.ones_like(ratios)
    loss, keep = compute_dis_loss(current, kl, advantages, mask, 0.3, 5.0)
    assert keep.tolist() == [False, True, True, True, False, False, False]
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    weights = torch.tensor([0, 0.7, 1, 6, 0, 0, 0], dtype=torch.float64)
    torch.testing.assert_close(current.grad, -weights * advantages.detach(), rtol=1e-14, atol=1e-14)
    assert kl.grad is None and advantages.grad is None
    assert torch.equal(mask, torch.ones_like(mask))


def test_dis_gradient_matches_finite_difference_of_ratio_surrogate():
    old = torch.tensor([-2.0, -3.0, -2.0, -2.0], dtype=torch.float64, requires_grad=True)
    current = torch.tensor([-1.7, -3.2, -3.0, 0.0], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([2.0, -3.0, 1.0, -1.0], dtype=torch.float64, requires_grad=True)
    loss, _ = compute_dis_loss(current, old - current, advantages, torch.ones(4), 0.3, 5.0)
    loss.sum().backward()

    def ratio_objective(logp, behavior, advantage):
        delta = logp - behavior
        return -math.exp(delta) * advantage if math.log(0.7) <= delta <= math.log(6) else 0.0

    eps = 1e-6
    expected = [
        (ratio_objective(c + eps, o, a) - ratio_objective(c - eps, o, a)) / (2 * eps)
        for c, o, a in zip(current.detach().tolist(), old.detach().tolist(), advantages.detach().tolist(), strict=True)
    ]
    torch.testing.assert_close(current.grad, torch.tensor(expected, dtype=torch.float64), rtol=1e-8, atol=1e-8)
    assert old.grad is None and advantages.grad is None


def test_dis_inactive_nonfinite_placeholders_and_original_action_denominators():
    current = torch.tensor([[-1.0, -1.0, math.nan, math.inf], [-1.0] * 4], requires_grad=True)
    kl = torch.tensor([[0.0, 1000, math.nan, -math.inf], [0.0, 0.0, 1000, -1000]])
    advantage = torch.tensor([[2.0, 2.0, math.nan, math.inf], [-2.0] * 4])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    loss, keep = compute_dis_loss(current, kl, advantage, mask, 0.3, 5.0)
    objective = (loss.sum(dim=1) / mask.sum(dim=1)).mean()
    objective.backward()
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(current.grad, torch.tensor([[-0.5, 0, 0, 0], [0.25, 0.25, 0, 0]]))
    assert keep.sum() == 3 and mask.sum() == 6


@pytest.mark.parametrize("zero_advantage", [False, True])
def test_dis_without_policy_signal_has_zero_gradient(zero_advantage):
    current = torch.tensor([-1.0, -2.0], requires_grad=True)
    advantage = torch.zeros(2) if zero_advantage else torch.tensor([1.0, -1.0])
    kl = torch.zeros(2) if zero_advantage else torch.tensor([1000.0, -1000.0])
    loss, keep = compute_dis_loss(current, kl, advantage, torch.ones(2), 0.3, 5.0)
    loss.sum().backward()
    assert torch.equal(loss, torch.zeros(2))
    assert torch.equal(current.grad, torch.zeros(2))
    assert not (keep & (advantage != 0)).any()


@pytest.mark.parametrize("field", ["current", "kl", "advantage"])
def test_dis_rejects_nonfinite_active_inputs(field):
    inputs = {name: torch.ones(2) for name in ("current", "kl", "advantage")}
    inputs[field][0] = math.nan
    with pytest.raises(RuntimeError, match="finite"):
        compute_dis_loss(inputs["current"], inputs["kl"], inputs["advantage"], torch.ones(2), 0.3, 5.0)


@pytest.mark.parametrize("low,high", [(-0.1, 5), (1.0, 5), (math.nan, 5), (0.3, -1), (0.3, math.inf)])
def test_dis_rejects_invalid_interval(low, high):
    with pytest.raises(ValueError, match="epsilon"):
        compute_dis_loss(torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1), low, high)


@pytest.mark.parametrize("mode", ["full", "skip"])
@pytest.mark.parametrize("gamma,policy_lambd,critic_lambd", [(1.0, 1.0, 1.0), (0.99, 0.95, 1.0), (0.7, 0.0, 0.4)])
def test_masked_gae_matches_independent_scalar_reference(mode, gamma, policy_lambd, critic_lambd):
    generator = torch.Generator().manual_seed(42)
    values = [torch.randn(length, generator=generator, dtype=torch.float64) for length in (1, 9, 257)]
    masks = [torch.tensor([1]), torch.tensor([0, 1, 1, 0, 0, 1, 0, 0, 0]), torch.arange(257) % 3 == 0]
    rewards = [2.0, -1.0, 0.75]
    actual = _targets(
        values,
        masks,
        rewards,
        observation_mode=mode,
        gamma=gamma,
        policy_lambd=policy_lambd,
        critic_lambd=critic_lambd,
    )
    for i, (value, mask, reward) in enumerate(zip(values, masks, rewards, strict=True)):
        expected = _reference(value.tolist(), mask.tolist(), reward, gamma, policy_lambd, critic_lambd, mode)
        for actual_list, expected_row in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_list[i], torch.tensor(expected_row, dtype=torch.float64), rtol=1e-10, atol=1e-10
            )


@pytest.mark.parametrize("mode", ["full", "skip"])
def test_new_modes_match_vanilla_when_every_token_is_an_action(mode):
    values = torch.tensor([[0.2, -0.4, 0.6]], dtype=torch.float64)
    rewards = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    expected = vanilla_gae(rewards, values, gamma=0.97, lambd=0.91)
    actual = _targets(
        [values[0]], [torch.ones(3)], [1.0], observation_mode=mode, gamma=0.97, policy_lambd=0.91, critic_lambd=0.91
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result[0], reference[0])


def test_skip_ignores_observation_values_and_full_skip_agree_at_unit_lambda():
    masks = [torch.tensor([1, 0, 0, 1, 0])]
    values = [torch.tensor([0.2, 9.0, -7.0, 0.4, 500.0], dtype=torch.float64)]
    changed = [torch.tensor([0.2, math.nan, math.inf, 0.4, math.nan], dtype=torch.float64)]
    skip = _targets(values, masks, [1.0])
    skip_changed = _targets(changed, masks, [1.0])
    for left, right in zip(skip, skip_changed, strict=True):
        torch.testing.assert_close(left[0], right[0])
    for mode in ("full", "skip"):
        advantages, returns = _targets(values, masks, [1.0], observation_mode=mode, policy_lambd=1.0)
        torch.testing.assert_close(advantages[0], torch.tensor([0.8, 0, 0, 0.6, 0], dtype=torch.float64))
        torch.testing.assert_close(returns[0], torch.tensor([1, 0, 0, 1, 0], dtype=torch.float64))
    full, _ = _targets(values, masks, [1.0], observation_mode="full")
    torch.testing.assert_close(full[0][0], torch.tensor(0.792925, dtype=torch.float64))
    torch.testing.assert_close(skip[0][0][0], torch.tensor(0.77, dtype=torch.float64))


@pytest.mark.parametrize("mode", ["full", "skip"])
def test_terminal_observations_are_excluded_and_critic_targets_are_independent(mode):
    values = [torch.tensor([0.2, 3.0, 0.4, math.nan, math.inf], requires_grad=True)]
    masks = [torch.tensor([1, 0, 1, 0, 0])]
    score = torch.tensor([1.0], requires_grad=True)
    first = _targets(values, masks, score, observation_mode=mode, policy_lambd=0.2, critic_lambd=0.8)
    second = _targets(values, masks, score, observation_mode=mode, policy_lambd=0.9, critic_lambd=0.8)
    assert not torch.equal(first[0][0], second[0][0])
    torch.testing.assert_close(first[1][0], second[1][0])
    assert all(not tensor.requires_grad for result in (*first, *second) for tensor in result)
    saved_returns = first[1][0].clone()
    first[0][0].add_(100)
    torch.testing.assert_close(first[1][0], saved_returns)
    assert torch.isnan(values[0][3]) and masks[0].tolist() == [1, 0, 1, 0, 0]


@pytest.mark.parametrize("mode", ["full", "skip"])
def test_adaptive_lambda_uses_action_count_in_both_modes(mode):
    values = [
        torch.tensor([0.2, 0.4], dtype=torch.float64),
        torch.tensor([0.1, 7.0, 0.2, 0.3, 8.0], dtype=torch.float64),
    ]
    masks = [torch.tensor([1, 1]), torch.tensor([1, 0, 1, 1, 0])]
    scores = [1.0, -1.0]
    actual = _targets(
        values,
        masks,
        scores,
        observation_mode=mode,
        policy_lambd=None,
        policy_lambda_mode="length-adaptive",
        policy_alpha=1.5,
    )
    for i, (value, mask, score) in enumerate(zip(values, masks, scores, strict=True)):
        lam = 1 - 1 / (1.5 * int(mask.sum()))
        expected = _reference(value.tolist(), mask.tolist(), score, 1.0, lam, 1.0, mode)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result[i], torch.tensor(reference, dtype=torch.float64))


def test_long_adaptive_trajectory_and_low_precision_values():
    length = 16384
    mask = torch.arange(length) % 2 == 0
    values = [torch.zeros(length, dtype=torch.float64)]
    advantages, returns = _targets(
        values, [mask], [1.0], policy_lambd=None, policy_lambda_mode="length-adaptive", policy_alpha=1.5
    )
    action_count = int(mask.sum())
    lam = 1 - 1 / (1.5 * action_count)
    expected = lam ** torch.arange(action_count - 1, -1, -1, dtype=torch.float64)
    torch.testing.assert_close(advantages[0][mask], expected, rtol=1e-10, atol=1e-10)
    assert torch.equal(returns[0][mask], torch.ones(action_count, dtype=torch.float64))
    promoted = _targets([torch.zeros(3, dtype=torch.bfloat16)], [torch.tensor([1, 0, 1])], [1.0])
    assert all(result[0].dtype == torch.float32 for result in promoted)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"observation_mode": "legacy"}, "observation_mode"),
        ({"gamma": math.nan}, "gamma"),
        ({"critic_lambd": 1.1}, "critic_lambd"),
        ({"policy_lambd": -0.1}, "policy_lambd"),
        ({"policy_lambda_mode": "length-adaptive", "policy_alpha": 0}, "policy_alpha"),
    ],
)
def test_invalid_gae_configuration(kwargs, error):
    with pytest.raises(ValueError, match=error):
        _targets([torch.zeros(2)], [torch.ones(2)], [1.0], **kwargs)


def test_masked_gae_rejects_empty_actions_nonbinary_masks_and_invalid_adaptive_length():
    for mask, message in ((torch.zeros(2), "at least one action"), (torch.tensor([1.0, 0.5]), "binary")):
        with pytest.raises(RuntimeError, match=message):
            _targets([torch.zeros(2)], [mask], [1.0])
    with pytest.raises(RuntimeError, match="alpha"):
        _targets(
            [torch.zeros(2)], [torch.tensor([1, 0])], [1.0], policy_lambda_mode="length-adaptive", policy_alpha=0.5
        )
    with pytest.raises(RuntimeError, match="finite values"):
        _targets([torch.tensor([math.nan])], [torch.ones(1)], [1.0])


@pytest.mark.parametrize("mode", ["full", "skip"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_long_fp32_targets_match_scalar_reference_with_accumulation_tolerance(mode, adaptive):
    values = torch.randn(16384, generator=torch.Generator().manual_seed(17))
    mask = torch.arange(16384) % 2 == 0
    lam = 1 - 1 / (1.5 * int(mask.sum())) if adaptive else 1.0
    actual = _targets(
        [values],
        [mask],
        [1.5],
        observation_mode=mode,
        policy_lambd=1.0,
        policy_lambda_mode="length-adaptive" if adaptive else "fixed",
        policy_alpha=1.5 if adaptive else None,
    )
    expected = _reference(values.tolist(), mask.tolist(), 1.5, 1.0, lam, 1.0, mode)
    for result, reference in zip(actual, expected, strict=True):
        # Includes float32 rounding of adaptive lambda as well as scan accumulation.
        torch.testing.assert_close(result[0], torch.tensor(reference), atol=2e-4, rtol=2e-4)


def test_scalar_tensor_reward_lists_are_detached_and_match_numeric_rewards():
    values = [torch.tensor([0.2, 0.4]), torch.tensor([0.1])]
    masks = [torch.ones(2), torch.ones(1)]
    rewards = [torch.tensor(1.0, requires_grad=True), torch.tensor(-1.0, requires_grad=True)]
    actual = _targets(values, masks, rewards)
    expected = _targets(values, masks, [1.0, -1.0])
    for results, references in zip(actual, expected, strict=True):
        for result, reference in zip(results, references, strict=True):
            torch.testing.assert_close(result, reference)
            assert not result.requires_grad
    with pytest.raises(ValueError, match="scalar tensors"):
        _targets(values, masks, [rewards[0], -1.0])

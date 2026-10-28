# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy loss dispatch with shared sampled-token inputs.

The kernels in :mod:`relax.utils.training.ppo_utils` take different argument
lists. These adapters normalise them to ``fn(args, *, log_probs, ppo_kl,
advantages)`` so the caller can look one up by name instead of branching on the
algorithm. Score centering additionally consumes generation-time candidates
through its explicit optional context; DIS requires the original generation
mask through ``loss_mask``.
"""

from typing import Any, Callable

import torch

from relax.algorithms.score_centering import compute_score_centering_loss
from relax.algorithms.spec import get_algorithm
from relax.utils.training.ppo_utils import (
    compute_cispo_loss,
    compute_dis_loss,
    compute_m2po_loss,
    compute_policy_loss,
    compute_rloo_loss,
    compute_sapo_loss,
)


def policy_loss_ppo_clip(
    args: Any,
    *,
    log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard clipped surrogate objective (GRPO, GSPO, PPO, REINFORCE++)."""
    return compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)


def policy_loss_sapo(
    args: Any,
    *,
    log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth trust region: sigmoid gating instead of a hard clip."""
    return compute_sapo_loss(
        ppo_kl=ppo_kl,
        advantages=advantages,
        tau_pos=getattr(args, "sapo_tau_pos", 1.0),
        tau_neg=getattr(args, "sapo_tau_neg", 1.05),
    )


def policy_loss_cispo(
    args: Any,
    *,
    log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clipped importance ratio that preserves the gradient direction."""
    return compute_cispo_loss(
        log_probs=log_probs,
        ppo_kl=ppo_kl,
        advantages=advantages,
        eps_clip=args.eps_clip,
        eps_clip_high=args.eps_clip_high,
    )


def policy_loss_m2po(
    args: Any,
    *,
    log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    """Adaptive asymmetric clipping from a harmful-token KL² budget."""
    return compute_m2po_loss(
        ppo_kl=ppo_kl,
        advantages=advantages,
        kl2_budget=args.m2po_kl2_budget,
        miniclip_low=args.m2po_miniclip_low,
        miniclip_high=args.m2po_miniclip_high,
    )


def policy_loss_rloo(
    args: Any,
    *,
    log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unclipped REINFORCE objective: ``-stopgrad(A) * log pi(y)``.

    ``ppo_kl`` is accepted and deliberately unused. Every other variant here
    corrects for the policy having moved since the rollout; this one has no
    such term, which is exactly why ``rloo`` declares
    ``requires_on_policy_updates`` -- the correction is missing from the maths,
    so it has to be guaranteed by the configuration instead.
    """
    return compute_rloo_loss(log_probs=log_probs, advantages=advantages)


def policy_loss_score_centering(
    args: Any,
    *,
    log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    score_centering_context: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Center the weighted score using the generation-time candidate
    distribution."""
    loss, metrics = compute_score_centering_loss(
        log_probs=log_probs,
        advantages=advantages,
        **score_centering_context,
        mode=args.score_centering_is,
        tis_clip=args.score_centering_tis_clip,
        mis_low=args.score_centering_mis_low,
        mis_high=args.score_centering_mis_high,
    )
    mask = score_centering_context["loss_mask"].to(loss.dtype)
    denominator = mask.sum().clamp_min(1)
    return loss, torch.zeros_like(loss), *((value * mask).sum() / denominator for value in metrics.values())


def policy_loss_dis(
    args: Any,
    *,
    log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DIS math adapter; original generation masks remain the loss denominator.

    The second result is a token-aligned drop indicator, not a normalized
    microbatch statistic. Training integration must separately aggregate kept
    policy-signal counts and prevent optimizer steps when that count is zero.
    """
    loss, keep = compute_dis_loss(
        log_probs, ppo_kl, advantages, loss_mask, args.dis_epsilon_low, args.dis_epsilon_high
    )
    return loss, (loss_mask.bool() & ~keep).float()


POLICY_LOSS_FNS: dict[str, Callable[..., tuple[Any, ...]]] = {
    "ppo_clip": policy_loss_ppo_clip,
    "sapo": policy_loss_sapo,
    "cispo": policy_loss_cispo,
    "m2po": policy_loss_m2po,
    "rloo": policy_loss_rloo,
    "score_centering": policy_loss_score_centering,
    "dis": policy_loss_dis,
}


def compute_policy_loss_for(
    args: Any,
    *,
    log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    score_centering_context: dict[str, torch.Tensor] | None = None,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch the registered policy loss and name its scalar diagnostics."""
    spec = get_algorithm(args.advantage_estimator)
    extra = {}
    if spec.policy_loss_fn == "score_centering":
        if score_centering_context is None:
            raise ValueError("Score centering requires rollout and current candidate log-probabilities")
        extra["score_centering_context"] = score_centering_context
    if spec.policy_loss_fn == "dis":
        if loss_mask is None:
            raise ValueError("DIS requires the original generation mask")
        extra["loss_mask"] = loss_mask
    pg_loss, pg_clipfrac, *scalar_metric_values = POLICY_LOSS_FNS[spec.policy_loss_fn](
        args,
        log_probs=log_probs,
        ppo_kl=ppo_kl,
        advantages=advantages,
        **extra,
    )
    if len(scalar_metric_values) != len(spec.policy_scalar_metric_names):
        raise ValueError(
            f"Policy loss {spec.policy_loss_fn!r} for algorithm {spec.name!r} returned "
            f"{len(scalar_metric_values)} scalar metrics, but its spec declares "
            f"{len(spec.policy_scalar_metric_names)}."
        )
    scalar_metrics: dict[str, torch.Tensor] = {}
    for name, value in zip(spec.policy_scalar_metric_names, scalar_metric_values, strict=True):
        value = torch.as_tensor(value, device=ppo_kl.device)
        if value.numel() != 1:
            raise ValueError(
                f"Policy loss {spec.policy_loss_fn!r} metric {name!r} must be scalar, got shape {value.shape}."
            )
        if value.is_complex():
            raise ValueError(f"Policy loss {spec.policy_loss_fn!r} metric {name!r} must be real-valued.")
        # loss.py stacks every metric with a float32 denominator. Normalising
        # here prevents one adapter's float64/int/bool diagnostic from
        # promoting the whole logging vector (and its distributed reduction).
        scalar_metrics[name] = value.reshape(()).to(dtype=torch.float32).detach()
    return pg_loss, pg_clipfrac, scalar_metrics

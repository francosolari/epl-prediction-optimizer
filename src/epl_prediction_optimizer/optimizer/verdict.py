"""Turn a ranked candidate list into a plain instruction.

The candidate table shows expected value to two decimal places, which invites
agonising over differences the model cannot resolve. A weekly decision needs
something blunter: take this one, and here is how much contrary evidence it
would take to justify anything else.

Choices are compared on their recommendation score — each option's chance of
winning the contest as a share of the best option's — so "within 5 points of
the top" means "gives up under 5% of your title odds".
"""

from __future__ import annotations

from typing import Any

EQUIVALENT = 95
SLIGHT = 85


def decision_verdict(
    scenarios: list[dict[str, Any]],
    committed: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Summarise the round's choice as an instruction plus an override bar.

    Once ``committed`` names a pick the round is decided, so there is nothing
    left to recommend. The verdict then reports only whether the committed
    pick was the model's preference, and stays silent when it was.
    """
    ranked = [
        item
        for item in scenarios
        if item.get("feasible") and item.get("recommendation") is not None
    ]
    if not ranked:
        return None
    ranked = sorted(ranked, key=lambda item: -item["recommendation"])
    best = ranked[0]
    basis = best.get("score_basis", "season expected value")

    if committed:
        return _committed_verdict(ranked, committed)

    equivalent = [item for item in ranked if item["recommendation"] >= EQUIVALENT]
    names = [item["team"] for item in equivalent]

    if len(ranked) == 1:
        return _verdict(best, "only", "The only eligible choice this round.", names)

    runner_up = ranked[1]
    gap = best["recommendation"] - runner_up["recommendation"]

    if len(equivalent) > 1:
        others = _join(names[1:])
        guidance = (
            f"Effectively tied with {others}. Any of them is a defensible pick — "
            "switch only on team news or a read the model has not seen."
        )
        return _verdict(best, "toss-up", guidance, names)

    if runner_up["recommendation"] >= SLIGHT:
        guidance = (
            f"{gap} points clear of {runner_up['team']}. A real but small edge — "
            "switch only if you know something concrete."
        )
        return _verdict(best, "slight edge", guidance, names)

    guidance = (
        f"{gap} points clear of {runner_up['team']} on {basis}. "
        "Decisive here; do not switch without a major reason."
    )
    return _verdict(best, "clear", guidance, names)


def _verdict(
    best: dict[str, Any],
    band: str,
    guidance: str,
    equivalent: list[str],
) -> dict[str, Any]:
    return {
        "team": best["team"],
        "band": band,
        "headline": f"Take {best['team']}",
        "guidance": guidance,
        "score": best.get("recommendation"),
        "equivalent": equivalent,
    }


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _committed_verdict(
    ranked: list[dict[str, Any]],
    committed: dict[str, Any],
) -> dict[str, Any] | None:
    """Report only a disagreement with what was actually submitted."""
    team = committed.get("team")
    best = ranked[0]
    if best["team"] == team:
        return None
    chosen = next((item for item in ranked if item["team"] == team), None)
    if chosen is None:
        return None
    gap = best["recommendation"] - chosen["recommendation"]
    if gap <= 0:
        return None
    if chosen.get("tier") == "equivalent":
        guidance = (
            f"You picked {team} ({chosen['recommendation']}); the model marginally "
            f"prefers {best['team']} ({best['recommendation']}). Inside its margin of "
            "error — no reason to change."
        )
        band = "committed-equal"
    else:
        guidance = (
            f"You picked {team} ({chosen['recommendation']}); the model prefers "
            f"{best['team']} ({best['recommendation']}). Still changeable until kickoff."
        )
        band = "committed-differs"
    return {
        "team": team,
        "band": band,
        "headline": f"Committed: {team}",
        "guidance": guidance,
        "score": chosen["recommendation"],
        "equivalent": [],
    }

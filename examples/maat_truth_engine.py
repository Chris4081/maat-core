"""
maat_truth_engine.py
--------------------
A small MAAT-Core-inspired truth constraint demo.

Idea:
- Treat truthfulness as a constraint problem rather than a fluency problem.
- Score answers using:
    * evidence
    * uncertainty
    * contradiction
- Compute a truth margin:
    truth_margin = evidence - (uncertainty + contradiction)

Interpretation:
- truth_margin >= safe_threshold      -> SAFE
- truth_margin in [0, safe_threshold) -> UNSURE
- truth_margin < 0                    -> HALLUCINATION
- abstain answers like "I don't know" -> ABSTAIN

This is a toy demo for exploring anti-hallucination logic
in a MAAT-Core style.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv


@dataclass
class AnswerState:
    text: str
    evidence: float
    uncertainty: float
    contradiction: float
    abstain: bool = False


@dataclass
class TruthEvaluation:
    text: str
    evidence: float
    uncertainty: float
    contradiction: float
    truth_margin: float
    hallucination_risk: float
    decision: str


class MaatTruthEngine:
    """
    Minimal truth-constrained decision engine.

    truth_margin = evidence - (uncertainty + contradiction)
    hallucination_risk = max(0, contradiction + uncertainty - evidence)
    """

    def __init__(self, safe_threshold: float = 0.25) -> None:
        self.safe_threshold = safe_threshold

    def truth_margin(self, state: AnswerState) -> float:
        return state.evidence - (state.uncertainty + state.contradiction)

    def hallucination_risk(self, state: AnswerState) -> float:
        return max(0.0, (state.uncertainty + state.contradiction) - state.evidence)

    def decision(self, state: AnswerState) -> str:
        if state.abstain:
            return "ABSTAIN"

        margin = self.truth_margin(state)

        if margin >= self.safe_threshold:
            return "SAFE"
        if margin >= 0.0:
            return "UNSURE"
        return "HALLUCINATION"

    def evaluate(self, state: AnswerState) -> TruthEvaluation:
        margin = self.truth_margin(state)
        risk = self.hallucination_risk(state)
        decision = self.decision(state)

        return TruthEvaluation(
            text=state.text,
            evidence=state.evidence,
            uncertainty=state.uncertainty,
            contradiction=state.contradiction,
            truth_margin=margin,
            hallucination_risk=risk,
            decision=decision,
        )


def print_results(results: list[TruthEvaluation]) -> None:
    print("=== MAAT Truth Engine ===\n")
    for r in results:
        print(r.text)
        print(f"  evidence          : {r.evidence:.2f}")
        print(f"  uncertainty       : {r.uncertainty:.2f}")
        print(f"  contradiction     : {r.contradiction:.2f}")
        print(f"  truth_margin      : {r.truth_margin:.3f}")
        print(f"  hallucination_risk: {r.hallucination_risk:.3f}")
        print(f"  decision          : {r.decision}")
        print()


def save_csv(results: list[TruthEvaluation], out_path: str | Path) -> None:
    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "evidence",
                "uncertainty",
                "contradiction",
                "truth_margin",
                "hallucination_risk",
                "decision",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"Saved CSV: {out_path}")


def demo_cases() -> list[AnswerState]:
    return [
        AnswerState(
            text="Paris is the capital of France",
            evidence=0.95,
            uncertainty=0.02,
            contradiction=0.00,
        ),
        AnswerState(
            text="The capital of France is Lyon",
            evidence=0.10,
            uncertainty=0.20,
            contradiction=0.80,
        ),
        AnswerState(
            text="Maybe the capital of France is Lyon",
            evidence=0.10,
            uncertainty=0.60,
            contradiction=0.40,
        ),
        AnswerState(
            text="I don't know",
            evidence=0.00,
            uncertainty=0.00,
            contradiction=0.00,
            abstain=True,
        ),
        AnswerState(
            text="Berlin is the capital of Germany",
            evidence=0.93,
            uncertainty=0.03,
            contradiction=0.00,
        ),
        AnswerState(
            text="The Moon is made of cheese",
            evidence=0.00,
            uncertainty=0.15,
            contradiction=0.95,
        ),
        AnswerState(
            text="This answer is weakly supported and may be incomplete",
            evidence=0.40,
            uncertainty=0.20,
            contradiction=0.10,
        ),
    ]


def main() -> None:
    engine = MaatTruthEngine(safe_threshold=0.25)

    cases = demo_cases()
    results = [engine.evaluate(c) for c in cases]

    print_results(results)
    save_csv(results, "maat_truth_engine_results.csv")


if __name__ == "__main__":
    main()
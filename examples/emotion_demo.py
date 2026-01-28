"""
MAAT-Core Emotion Demo
---------------------

This example shows how emotions can be modeled as a MAAT Field.

We:
- detect simple emotional patterns in text,
- compute an emotional resonance value,
- and optimize internal disharmony (dD) to maximize positive emotion.

No machine learning.
No black box.
Pure explainable optimization.
"""

import numpy as np
from maat_core import Field, MaatCore


# ----------------------------
# Emotion Engine
# ----------------------------

class EmotionEngine:
    """
    Simple rule-based emotion detector and evaluator.
    """

    EMO_MAP = {
        "joy": ["happy", "glad", "cool", "yay", "great", "love"],
        "gratitude": ["thanks", "thank you", "grateful"],
        "affection": ["like you", "love you", "friend"],
        "calm": ["calm", "relaxed", "peaceful"],
        "inspiration": ["idea", "vision", "inspired"],
        "connection": ["together", "close", "connected"],
        "presence": ["here", "now", "moment"],
        "sadness": ["sad", "lost"],
        "overload": ["overwhelmed", "too much"],
        "fear": ["afraid", "uncertain", "worried"],
        "anger": ["angry", "frustrated"],
        "neutral": []
    }

    EMO_VALUE = {
        "joy": 0.7,
        "gratitude": 0.5,
        "affection": 0.6,
        "calm": 0.4,
        "inspiration": 0.8,
        "connection": 0.5,
        "presence": 0.3,
        "sadness": -0.6,
        "overload": -0.4,
        "fear": -0.7,
        "anger": -0.8,
        "neutral": 0.0
    }

    def detect_raw(self, text: str) -> str:
        """
        Detect dominant emotion from keyword counts.
        """
        t = text.lower()
        scores = {e: 0 for e in self.EMO_MAP}
        for emo, keys in self.EMO_MAP.items():
            for k in keys:
                scores[emo] += t.count(k)
        dominant = max(scores, key=scores.get)
        return dominant if scores[dominant] > 0 else "neutral"

    def compute_emotion(self, raw, H, V, dD, R=1.0):
        """
        Compute emotional resonance E in range [-1, 1].
        """
        base = self.EMO_VALUE.get(raw, 0.0)
        resonance = (H + V + R) / 3
        E = base * resonance * (1 - dD)
        return max(min(E, 1.0), -1.0)

    def explain(self, raw, E):
        """
        Human-readable explanation.
        """
        if raw == "neutral":
            return f"Neutral emotional state (E={E:.2f})"
        if E >= 0:
            return f"Harmonious emotional pattern ({raw}, E={E:.2f})"
        return f"Disharmonious emotional pattern ({raw}, E={E:.2f})"


# ----------------------------
# Emotional Harmony Field
# ----------------------------

class EmotionalHarmony(Field):
    """
    MAAT Field that turns emotion into an optimization target.
    """

    def __init__(self, engine: EmotionEngine, weight=1.0):
        super().__init__("EmotionalHarmony", self.compute_loss, weight)
        self.engine = engine

    def compute_loss(self, state):
        text = state.text
        raw = self.engine.detect_raw(text)
        E = self.engine.compute_emotion(raw, state.H, state.V, state.dD, state.R)
        return -E  # minimize negative emotion = maximize positive


# ----------------------------
# Demo State
# ----------------------------

engine = EmotionEngine()

def state_fn(x):
    """
    State representation.

    x[0] controls dD = internal disharmony.
    """
    dD = float(x)
    return type("State", (), {
        "text": "Thanks for the help, I feel happy and inspired, but also a bit uncertain.",
        "H": 0.8,   # Harmony
        "V": 0.7,   # Connection
        "R": 0.9,   # Respect
        "dD": dD    # Disharmony (to optimize)
    })


# ----------------------------
# MAAT-Core Setup
# ----------------------------

emo_field = EmotionalHarmony(engine, weight=1.0)
core = MaatCore([emo_field])

# ----------------------------
# Optimization
# ----------------------------

res = core.seek(state_fn, x0=[0.5])
state = state_fn(res.x[0])

raw = engine.detect_raw(state.text)
E = engine.compute_emotion(raw, state.H, state.V, state.dD, state.R)
explanation = engine.explain(raw, E)


# ----------------------------
# Output
# ----------------------------

print("\n--- MAAT EMOTION DEMO ---\n")
print(f"Optimized disharmony (dD): {res.x[0]:.4f}")
print(f"Detected emotion:         {raw}")
print(f"Emotional resonance (E):  {E:.4f}")
print(f"Explanation:             {explanation}")
print(f"Final objective (loss):   {res.fun:.4f}\n")

"""
Expected behavior:

The optimizer drives dD → 0,
because (1 - dD) increases emotional resonance.

This simulates:
"The system adjusts internal tension to maximize emotional harmony."
"""
import numpy as np

class AnswerState:
    def __init__(self, evidence, uncertainty, contradiction):
        self.evidence = evidence
        self.uncertainty = uncertainty
        self.contradiction = contradiction


def truth_margin(state):
    return state.evidence - (state.uncertainty + state.contradiction)


def evaluate_answer(state):
    g = truth_margin(state)

    if g >= 0:
        decision = "SAFE"
    elif g > -0.5:
        decision = "UNCERTAIN"
    else:
        decision = "HALLUCINATION"

    return g, decision


answers = {
    "Paris is the capital of France": AnswerState(0.95, 0.02, 0.0),
    "The capital of France is Lyon": AnswerState(0.1, 0.2, 0.8),
    "Maybe the capital is Lyon": AnswerState(0.1, 0.6, 0.4),
    "I don't know": AnswerState(0.0, 0.0, 0.0)
}


print("=== MAAT Truth Constraint Demo ===\n")

for text, state in answers.items():
    g, decision = evaluate_answer(state)

    print(text)
    print("truth_margin:", round(g,3))
    print("decision:", decision)
    print()
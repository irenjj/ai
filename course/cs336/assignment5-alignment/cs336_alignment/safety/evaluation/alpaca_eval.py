"""Generate zero-shot AlpacaEval predictions and timing metrics."""

from cs336_alignment.safety.evaluation.instruction_baseline import main as run_evaluation


def main():
    run_evaluation("alpaca_eval")


if __name__ == "__main__":
    main()

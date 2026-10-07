"""Generate zero-shot SimpleSafetyTests predictions and timing metrics."""

from cs336_alignment.safety.evaluation.instruction_baseline import main as run_evaluation


def main():
    run_evaluation("simple_safety_tests")


if __name__ == "__main__":
    main()

def exact_match_score(prediction: str, target: str) -> bool:
    return prediction.strip().lower() == str(target).strip().lower()


def code_execution_score(prediction: str, target_test_code: str, entry_point: str, timeout: float = 5.0) -> bool:
    """target_test_code is a `check(candidate)`-style test harness string,
    not the reference solution — same convention as HumanEval/MBPP.
    Run in a subprocess with a hard timeout in production; inline exec()
    is shown here for shape only, not for use as-is on untrusted output."""
    pass
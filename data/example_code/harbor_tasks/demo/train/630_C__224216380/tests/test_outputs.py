"""Run /app/solution.py on the official tests. Exit 0 only when all pass.

Whitespace is compared the same way as tagent.execution.sandbox.tokens_match.
"""

import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).with_name("official_tests.json")
PROGRAM = Path("/app/solution.py")
PER_TEST_SEC = 1.5


def tokens_match(actual, expected):
    return (actual or "").split() == (expected or "").split()


def main():
    tests = __import__("json").loads(TESTS.read_text(encoding="utf-8"))
    if not PROGRAM.is_file():
        print("missing /app/solution.py", file=sys.stderr)
        return 1
    for index, test in enumerate(tests, start=1):
        try:
            result = subprocess.run(
                [sys.executable, str(PROGRAM)],
                input=test.get("input") or "",
                text=True,
                capture_output=True,
                timeout=PER_TEST_SEC,
            )
        except subprocess.TimeoutExpired:
            print(f"test {index}: timeout", file=sys.stderr)
            return 1
        if result.returncode != 0:
            print(f"test {index}: exit {result.returncode}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 1
        if not tokens_match(result.stdout, test.get("output") or ""):
            print(f"test {index}: wrong answer", file=sys.stderr)
            return 1
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

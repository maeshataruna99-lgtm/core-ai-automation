"""Find secrets before they enter Git: python scripts/check_secrets.py [--staged].

Why: A key that reaches a public repository must be treated as stolen, so the check has to run
before the commit. Output names the file, line, and rule, but never the value, because terminal
output gets shared.

Used by: `.githooks/pre-commit` with `--staged`, `.github/workflows/secrets.yml`, and
`tests/test_secrets.py`.

Needs: Python and Git, with no other packages, so that it works before the project is installed.
Does: scans staged files with `--staged`, or else all files that Git tracks or would add.
Prints: one line per finding, without the secret.
Exit code: 0 if clean, 1 if something was found or the check could not run.
Changes: nothing. It reads the local `.env` only to look for its exact values elsewhere.
"""

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

# Each pattern is a known secret format, or a secret-looking name assigned a literal value.
PATTERNS = {
    "API key (sk- format)": re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,})"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "Bearer token": re.compile(r"\bBearer\s+([A-Za-z0-9._~+/-]{20,})"),
    "Secret assigned a value": re.compile(
        r"(?i)\b[\w-]*(?:api[_-]?key|secret|token|password|passwd)(?!s\b)[\w-]*[\"']?\s*[:=]\s*[\"']?"
        r"([A-Za-z0-9_/+-]{16,})(?![\w.(])"
    ),
}
# Words that mark a value as an obvious placeholder or test value.
HARMLESS_WORDS = ("test", "example", "fake", "dummy", "placeholder", "sample", "secret_value")
# Code that reads a key, such as `api_key=settings.llm_api_key.get_secret_value()`, is not a value.
CODE_LOOKING = re.compile(r"^(?:settings|self|os|config|env|args|request)\.|\(")
BLOCKED_NAME = re.compile(
    r"(?:^|/)(?:\.env(?:\.(?!example$)[^/]+)?|[^/]+\.env)$|\.(?:pem|key|p12|pfx|db|sqlite3?)$",
    re.IGNORECASE,
)
# Own files whose job is to describe these patterns.
SKIPPED_FILES = {"scripts/check_secrets.py"}
MIN_LOCAL_VALUE_LENGTH = 8


def run_git(*arguments: str) -> str:
    """Run Git and return its output.

    Why: Every list of files comes from Git, so ignored files such as `.env` are not reported
    unless they are actually about to be committed.

    Used by: `files_to_scan` and `read_file`.

    Args:
        *arguments: The Git arguments.

    Returns:
        The output.

    Raises:
        subprocess.CalledProcessError: Git failed.
    """
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def files_to_scan(staged: bool) -> list[str]:
    """List the files to check.

    Why: `--staged` looks at exactly what the commit would contain. Otherwise this covers what Git
    tracks, plus new files that `git add .` would pick up.

    Used by: `main`.

    Args:
        staged: Whether to look only at staged files.

    Returns:
        The file paths, relative to the repository root.
    """
    if staged:
        output = run_git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    else:
        output = run_git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return [name for name in output.split("\0") if name]


def read_file(name: str, staged: bool) -> str | None:
    """Read a file's text, as it would be committed.

    Why: With `--staged`, the staged version is the one that matters, because the file on disk may
    already have been fixed while the commit still holds the secret.

    Used by: `main`.

    Args:
        name: The file path.
        staged: Whether to read the staged version.

    Returns:
        The text, or None for a binary or missing file.
    """
    try:
        if staged:
            data = subprocess.run(
                ["git", "show", f":{name}"], capture_output=True, check=True
            ).stdout
        else:
            data = Path(name).read_bytes()
    except (OSError, subprocess.CalledProcessError):
        return None
    if b"\0" in data:
        return None
    return data.decode("utf-8", "replace")


def local_secret_values() -> set[str]:
    """Read the values of secret settings in the local `.env`.

    Why: Patterns cannot know every vendor's format, but this file holds this project's real keys,
    so an exact match is certain. Values stay in memory, and are never printed.

    Used by: `main`.

    Returns:
        The values that are long enough to compare safely.
    """
    try:
        lines = Path(".env").read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return set()
    values = set()
    for line in lines:
        name, separator, value = line.partition("=")
        if separator and re.search(r"(?i)key|secret|token|password", name):
            value = value.strip().strip("\"'")
            if len(value) >= MIN_LOCAL_VALUE_LENGTH:
                values.add(value)
    return values


def looks_harmless(value: str) -> bool:
    """Decide whether a matched value is plainly not a real secret.

    Why: A guard that blocks normal work gets switched off, so test values and code need to pass.

    Used by: `scan_text`.

    Args:
        value: The matched value.

    Returns:
        True if it is a placeholder, a test value, or code.
    """
    lowered = value.lower()
    return any(word in lowered for word in HARMLESS_WORDS) or bool(CODE_LOOKING.search(value))


def scan_text(text: str, local_values: set[str]) -> list[tuple[int, str]]:
    """Find secrets in a text.

    Why: This returns only the line and the rule, so a value cannot reach the output by accident.

    Used by: `main`.

    Args:
        text: The file's text.
        local_values: Exact values to look for.

    Returns:
        One line number and rule name per finding.
    """
    findings = []
    for number, line in enumerate(text.splitlines(), start=1):
        if any(value in line for value in local_values):
            findings.append((number, "Value from your local .env file"))
            continue
        for rule, pattern in PATTERNS.items():
            match = pattern.search(line)
            if match and not looks_harmless(match.group(match.lastindex or 0)):
                findings.append((number, rule))
                break
    return findings


def main() -> int:
    """Scan the files and report the result.

    Why: If the check cannot run, it reports failure. A guard that silently passes when broken would
    be worse than none.

    Used by: the `__main__` block below.

    Returns:
        0 if clean, 1 if a secret was found or the check could not run.
    """
    staged = "--staged" in sys.argv[1:]
    try:
        names = files_to_scan(staged)
    except (OSError, subprocess.CalledProcessError):
        print("Secret check could not run: Git failed.")
        return 1

    local_values = local_secret_values()
    findings = []
    for name in names:
        posix = str(PurePosixPath(name))
        if BLOCKED_NAME.search(posix):
            findings.append(f"{name}: this kind of file must never be committed")
            continue
        if posix in SKIPPED_FILES:
            continue
        text = read_file(name, staged)
        if text is not None:
            findings.extend(
                f"{name}:{line}: {rule}" for line, rule in scan_text(text, local_values)
            )

    if not findings:
        print(f"Secret check passed: {len(names)} files.")
        return 0
    print("Secret check FAILED. Values are not shown.")
    print("\n".join(f"  {finding}" for finding in findings))
    print("Remove the secret. If it was ever pushed or shared, revoke it and create a new one.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

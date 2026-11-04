"""Tests that secrets cannot be committed, and do not leak at runtime.

Why: A key that reaches a public repository must be treated as stolen. These tests prove that the
guard really blocks a commit, and they follow a key through the real app and evaluation command.
Fake keys are built at runtime, so this file does not trip the scanners it is testing.

Used by: `pytest`, locally and in `.github/workflows/ci.yml`.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ai_ops.config import Settings
from ai_ops.llm.factory import build_provider

REPO = Path(__file__).parent.parent
SCANNER = REPO / "scripts" / "check_secrets.py"
# Built from parts, so that the scanner does not flag this file.
FAKE_DEEPSEEK_KEY = "sk-" + "4f3a9c1e7b2d" * 3
FAKE_GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5" * 4
FAKE_VALUE = "a1b2c3d4e5" * 2
FAKE_PASSWORD = "correcthorse" + "batterystaple"
FAKE_PRIVATE_KEY = "-----BEGIN RSA " + "PRIVATE KEY-----"
needs_git = pytest.mark.skipif(not (REPO / ".git").exists(), reason="needs a Git checkout")


def git(directory: Path, *arguments: str) -> subprocess.CompletedProcess:
    """Run Git in a directory.

    Why: Each scanner test uses a real, separate repository.

    Used by: the tests in this module.

    Args:
        directory: The repository.
        *arguments: The Git arguments.

    Returns:
        The finished process.
    """
    return subprocess.run(["git", *arguments], cwd=directory, capture_output=True, text=True)


def scan(directory: Path, *arguments: str) -> subprocess.CompletedProcess:
    """Run the secret scanner in a directory.

    Why: This tests it the way the commit hook runs it.

    Used by: the tests in this module.

    Args:
        directory: The repository.
        *arguments: The scanner's arguments.

    Returns:
        The finished process.
    """
    return subprocess.run(
        [sys.executable, str(SCANNER), *arguments], cwd=directory, capture_output=True, text=True
    )


@pytest.fixture
def repository(tmp_path):
    """Provide an empty Git repository.

    Why: Scanner tests stage files, which must never touch the real repository.

    Used by: the scanner tests in this module.

    Args:
        tmp_path: Pytest's temporary directory.

    Returns:
        The repository's path.
    """
    git(tmp_path, "init", "-q")
    return tmp_path


def stage(repository: Path, name: str, content: str) -> None:
    """Write a file and stage it.

    Why: The commit hook looks at exactly what is staged.

    Used by: the scanner tests in this module.

    Args:
        repository: The repository.
        name: The file name.
        content: The content.
    """
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    assert git(repository, "add", "-f", name).returncode == 0


@pytest.mark.parametrize(
    ("name", "content"),
    [
        pytest.param("config.py", f'KEY = "{FAKE_DEEPSEEK_KEY}"\n', id="deepseek-style-key"),
        pytest.param("notes.md", f"token {FAKE_GITHUB_TOKEN}\n", id="github-token"),
        pytest.param("id_rsa.txt", f"{FAKE_PRIVATE_KEY}\nabc\n", id="private-key"),
        pytest.param("db.yml", f"password: {FAKE_PASSWORD}\n", id="password-with-no-digits"),
        pytest.param("settings.yml", f"api_key: {FAKE_VALUE}\n", id="key-in-yaml"),
        pytest.param("run.sh", f"export AI_OPS_LLM_API_KEY={FAKE_VALUE}\n", id="key-in-script"),
        pytest.param(
            "client.py",
            f'headers = {{"Authorization": "Bearer {FAKE_VALUE}"}}\n',
            id="bearer-token",
        ),
        pytest.param("deep/down/file.json", f'{{"k": "{FAKE_DEEPSEEK_KEY}"}}\n', id="nested-file"),
    ],
)
def test_scanner_blocks_staged_secrets(repository, name, content):
    """Check each kind of secret in a staged file.

    Why: This is the guard that stops a key before it enters history, so it must really fail. It
    also must not print the secret, since terminal output gets shared.

    Used by: pytest.

    Args:
        repository: The test repository.
        name: The file name.
        content: The content.
    """
    stage(repository, name, content)

    result = scan(repository, "--staged")

    assert result.returncode == 1
    assert name in result.stdout
    assert "a1b2c3d4e5f6" not in result.stdout
    assert FAKE_DEEPSEEK_KEY not in result.stdout
    assert FAKE_GITHUB_TOKEN not in result.stdout


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        "config/.env.production",
        "server.pem",
        "private.key",
        "cert.p12",
        "app.db",
    ],
)
def test_scanner_blocks_files_that_should_never_be_committed(repository, name):
    """Check file names, whatever is in the file.

    Why: A `.env` file is secret by its nature, even if its content looks harmless today.

    Used by: pytest.

    Args:
        repository: The test repository.
        name: The file name.
    """
    stage(repository, name, "harmless\n")

    assert scan(repository, "--staged").returncode == 1


@pytest.mark.parametrize(
    ("name", "content"),
    [
        pytest.param(".env.example", "AI_OPS_LLM_API_KEY=\n", id="env-example-with-no-value"),
        pytest.param("a.py", 'API_KEY = "sk-test-secret-key"\n', id="obvious-test-value"),
        pytest.param(
            "b.py", "api_key=settings.llm_api_key.get_secret_value()\n", id="code-reading-a-key"
        ),
        pytest.param("c.md", "Set `AI_OPS_LLM_API_KEY` in your env file.\n", id="documentation"),
        pytest.param(
            "d.py", 'headers={"Authorization": f"Bearer {api_key}"}\n', id="template-not-a-value"
        ),
    ],
)
def test_scanner_allows_harmless_text(repository, name, content):
    """Check things that look similar but are safe.

    Why: A guard that blocks normal work gets switched off.

    Used by: pytest.

    Args:
        repository: The test repository.
        name: The file name.
        content: The content.
    """
    stage(repository, name, content)

    result = scan(repository, "--staged")

    assert result.returncode == 0, result.stdout


def test_scanner_catches_the_real_value_from_the_local_env_file(repository):
    """Check a key with an unusual format, using the local env file.

    Why: Patterns cannot know each vendor's format, but the scanner knows this project's real keys.
    It must find them anywhere, and it must not print them.

    Used by: pytest.

    Args:
        repository: The test repository.
    """
    value = "unusual_format_" + "9" * 12
    (repository / ".env").write_text(f"AI_OPS_LLM_API_KEY={value}\n", encoding="utf-8")
    stage(repository, "notes.md", f"my key is {value}\n")

    result = scan(repository, "--staged")

    assert result.returncode == 1
    assert value not in result.stdout


def test_scanner_fails_when_it_cannot_run(tmp_path):
    """Check the scanner outside a Git repository, where Git itself fails.

    Why: A guard that passes when it is broken gives false safety, so not being able to check must
    count as a failure.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory, which is not a repository.
    """
    result = scan(tmp_path)

    assert result.returncode == 1
    assert "could not run" in result.stdout


def test_commit_hook_really_blocks_a_commit(repository):
    """Run a real `git commit` with the project's hook.

    Why: A scanner that works is no use if the hook does not call it.

    Used by: pytest.

    Args:
        repository: The test repository.
    """
    hooks = repository / "hooks"
    hooks.mkdir()
    hook = hooks / "pre-commit"
    shutil.copy(REPO / ".githooks" / "pre-commit", hook)
    hook.chmod(0o755)
    scripts = repository / "scripts"
    scripts.mkdir()
    shutil.copy(SCANNER, scripts / "check_secrets.py")
    git(repository, "config", "core.hooksPath", "hooks")
    git(repository, "config", "user.email", "test@example.test")
    git(repository, "config", "user.name", "Test")
    stage(repository, "config.py", f'KEY = "{FAKE_DEEPSEEK_KEY}"\n')

    blocked = git(repository, "commit", "-m", "add key")

    assert blocked.returncode != 0
    assert git(repository, "log", "--oneline").stdout == ""
    assert FAKE_DEEPSEEK_KEY not in blocked.stdout + blocked.stderr


@needs_git
def test_this_repository_has_no_secrets():
    """Run the scanner on all files in this repository.

    Why: This is the same check as the hook, and it also covers a commit made with the hook off.

    Used by: pytest.
    """
    result = scan(REPO)

    assert result.returncode == 0, result.stdout


@needs_git
@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        ".env.production",
        "local.env",
        "key.pem",
        "my.key",
        "ai_ops.db",
        "data/ai_ops.db",
        "data/uploads/private.md",
    ],
)
def test_secret_files_are_ignored_by_git(name):
    """Check the ignore rules.

    Why: A plain `git add .` must not be able to pick these up.

    Used by: pytest.

    Args:
        name: The file name.
    """
    assert git(REPO, "check-ignore", "-q", "--no-index", name).returncode == 0


@needs_git
def test_env_example_is_not_ignored_and_holds_no_values():
    """Check the template.

    Why: It must be committed for people to find, so it must hold no value in any secret variable.

    Used by: pytest.
    """
    assert git(REPO, "check-ignore", "-q", "--no-index", ".env.example").returncode != 0
    lines = (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("AI_OPS_") and "API_KEY" in line:
            assert line.split("=", 1)[1] == ""


def test_env_example_covers_the_deepseek_setup():
    """Check that the template works as a starting point.

    Why: Copying the file and adding one key should be all that DeepSeek needs.

    Used by: pytest.
    """
    text = (REPO / ".env.example").read_text(encoding="utf-8")

    assert "AI_OPS_LLM_PROVIDER=deepseek" in text
    assert "AI_OPS_LLM_API_KEY=" in text


def deepseek_settings(**overrides) -> dict:
    """Build the environment for a DeepSeek run with a fake key and an unreachable server.

    Why: Port 1 refuses connections at once, so the failure path runs offline and costs nothing.

    Used by: the leak tests in this module.

    Args:
        **overrides: Extra environment variables.

    Returns:
        The environment.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("AI_OPS_")}
    env.update(
        AI_OPS_ENV_FILE="",
        AI_OPS_LLM_PROVIDER="deepseek",
        AI_OPS_LLM_API_KEY=FAKE_DEEPSEEK_KEY,
        AI_OPS_LLM_BASE_URL="http://127.0.0.1:1",
        AI_OPS_LLM_MAX_RETRIES="0",
        AI_OPS_KNOWLEDGE_DIR=str(REPO / "data" / "knowledge"),
        **overrides,
    )
    return env


def test_failed_evaluation_never_prints_or_stores_the_key():
    """Follow the key through a real command that fails.

    Why: A connection error is where a key is most likely to leak, because libraries like to print
    the request. Everything the command prints or writes is checked.

    Used by: pytest.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_ops.evaluation",
            "--golden",
            str(REPO / "data" / "evaluation" / "golden.json"),
            "--output-dir",
            "runs",
        ],
        env=deepseek_settings(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert FAKE_DEEPSEEK_KEY not in result.stdout + result.stderr
    for path in Path(".").rglob("*"):
        if path.is_file():
            assert FAKE_DEEPSEEK_KEY not in path.read_text(encoding="utf-8", errors="ignore")


def test_key_is_hidden_when_printing_settings_and_provider():
    """Check the ways a key could reach a log by accident.

    Why: A debugging `print` or a log line with these objects must not reveal the key.

    Used by: pytest.
    """
    settings = Settings(llm_provider="deepseek", llm_api_key=FAKE_DEEPSEEK_KEY)

    assert FAKE_DEEPSEEK_KEY not in repr(settings)
    assert FAKE_DEEPSEEK_KEY not in str(settings)
    assert FAKE_DEEPSEEK_KEY not in settings.model_dump_json()
    assert FAKE_DEEPSEEK_KEY not in repr(build_provider(settings))


def test_health_and_api_docs_never_show_the_key(tmp_path):
    """Check the public endpoints of an app holding a key.

    Why: The API has no login, so nothing it serves may contain a secret.

    Used by: pytest.

    Args:
        tmp_path: Pytest's temporary directory.
    """
    from fastapi.testclient import TestClient

    from ai_ops.api.app import create_app
    from ai_ops.llm.mock import MockProvider

    settings = Settings(
        llm_provider="deepseek",
        llm_api_key=FAKE_DEEPSEEK_KEY,
        knowledge_dir=tmp_path / "none",
        database_path=tmp_path / "app.db",
    )
    with TestClient(create_app(settings, llm_provider=MockProvider())) as client:
        for path in ("/health", "/openapi.json", "/approvals", "/ui/", "/ui/app.js"):
            assert FAKE_DEEPSEEK_KEY not in client.get(path).text, path


def test_committed_evaluation_reports_hold_only_known_fields():
    """Check the reports stored in the repository.

    Why: These go into Git, so they must hold only an explicit list of settings.

    Used by: pytest.
    """
    for path in (REPO / "docs" / "evaluation" / "runs").glob("*.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        assert not [name for name in metadata if "key" in name.lower() or "secret" in name.lower()]

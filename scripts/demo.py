"""Run the whole demo with one command: python scripts/demo.py.

Why: A reviewer should see the project work in a minute, with no setup beyond installing it. The
script also checks each result, so the demo doubles as an end-to-end test. Each claim in it is
something that really happened in that run.

Used by: people, `tests/test_demo_and_docs.py`, and the demo recording in `docs/demo/`.

Needs: the project installed (see the README). It needs no API key and no network.
Does: starts the API with the mock provider on a free local port, calls it, then stops it.
Prints: each step and its result. The mock provider echoes questions, so answers are not real.
Exit code: 0 if every check passed, 1 if a check failed or the server did not start.
Changes: nothing in the repository. The database, log, and report go in a temporary directory.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
LINE = "-" * 72


class DemoError(Exception):
    """A step did not give the expected result.

    Why: One type lets `main` print a short message and exit with code 1.

    Used by: `check` and `Server`.
    """


def check(condition: bool, message: str) -> None:
    """Stop the demo if a result is wrong.

    Why: A demo that keeps going after a failure could show something misleading. `assert` is not
    used because Python can be told to skip it.

    Used by: the steps in `run_steps`.

    Args:
        condition: Whether the result is right.
        message: What was expected.

    Raises:
        DemoError: The condition is false.
    """
    if not condition:
        raise DemoError(message)


def step(number: int, title: str) -> None:
    """Print a step heading.

    Why: Makes the output readable when it is recorded for the demo.

    Used by: `run_steps`.

    Args:
        number: The step number.
        title: What the step shows.
    """
    print(f"\n{LINE}\n{number}. {title}\n{LINE}")


def show(label: str, value: Any) -> None:
    """Print one result line.

    Why: Keeps every result in the same format.

    Used by: `run_steps`.

    Args:
        label: What the value is.
        value: The value. Anything except text is printed as JSON.
    """
    print(f"  {label:<12} {value if isinstance(value, str) else json.dumps(value)}")


class Server:
    """Start the real API as a separate process, and stop it at the end.

    Why: The demo calls the same command a person would run, over real HTTP. Logs go to a file, not
    a pipe, because a full pipe that nothing is reading would block the server.

    Used by: `main`.
    """

    def __init__(self, directory: Path) -> None:
        """Choose a free port and prepare the environment.

        Why: A fixed port could already be in use. Inherited `AI_OPS_*` variables are removed, so
        the demo always uses the mock provider and never contacts a real service.

        Used by: `main`.

        Args:
            directory: The temporary directory for the database and the log.
        """
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = directory / "server.log"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("AI_OPS_")}
        self.env.update(
            # Ignore a local .env, so a real key or messaging service can never be used here.
            AI_OPS_ENV_FILE="",
            AI_OPS_LLM_PROVIDER="mock",
            AI_OPS_PORT=str(self.port),
            AI_OPS_KNOWLEDGE_DIR=str(REPO / "data" / "knowledge"),
            AI_OPS_DATABASE_PATH=str(directory / "demo.db"),
            AI_OPS_UPLOADS_DIR=str(directory / "uploads"),
        )
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def __enter__(self) -> "Server":
        """Start the process and wait until it answers.

        Why: Calling it before it is ready would look like a failure.

        Used by: the `with` statement in `main`.

        Returns:
            This object.

        Raises:
            DemoError: The server did not start within 30 seconds.
        """
        self.log_file = self.log_path.open("w", encoding="utf-8")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [sys.executable, "-m", "ai_ops"],
            env=self.env,
            stderr=self.log_file,
            stdout=subprocess.DEVNULL,
            creationflags=flags,
        )
        for _ in range(150):
            try:
                urllib.request.urlopen(f"{self.base_url}/health", timeout=1).close()
                return self
            except OSError:
                time.sleep(0.2)
        self.__exit__()
        raise DemoError("The server did not start. See the log for details.")

    def __exit__(self, *_: object) -> None:
        """Stop the process, even if a step failed.

        Why: A leftover server would keep its port and its files.

        Used by: the `with` statement in `main`.

        Args:
            *_: Exception details, which are not used.
        """
        stop = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
        self.process.send_signal(stop)
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log_file.close()

    def call(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        """Send one HTTP request.

        Why: Uses only the standard library, and treats error statuses as normal results, since
        the demo shows some of them on purpose.

        Used by: `run_steps`.

        Args:
            path: The URL path.
            body: The JSON body. None sends a GET.

        Returns:
            The status code and the parsed response.
        """
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}", data, {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def tool_log_lines(self) -> list[dict]:
        """Read the tool calls from the server's log.

        Why: Shows the audit trail that the API keeps for each tool call.

        Used by: `run_steps`.

        Returns:
            The log entries for tool calls.
        """
        self.log_file.flush()
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.startswith("{")]
        return [entry for entry in entries if entry["message"] == "tool call"]


def run_steps(server: Server, directory: Path) -> None:
    """Walk through the demo sequence.

    Why: The order follows `docs/PORTFOLIO_POSITIONING.md`, so that a reviewer sees each main
    feature once.

    Used by: `main`.

    Args:
        server: The running server.
        directory: Where the evaluation report goes.

    Raises:
        DemoError: A step gave the wrong result.
    """
    step(1, "The API is running, and the knowledge base is loaded")
    status, health = server.call("/health")
    check(status == 200 and health["documents"] > 0, "Expected documents to be loaded")
    show("GET /health", health)

    step(2, "Ask a question: retrieval finds the source")
    question = "How long do card refunds take?"
    status, answer = server.call("/ask", {"question": question})
    check(status == 200 and answer["found"], "Expected a relevant source")
    check(answer["sources"][0]["id"] == "refund-policy.md#0", "Expected the refund policy first")
    show("question", question)
    show("sources", [source["id"] for source in answer["sources"]])
    show("answer", answer["answer"])
    show("note", "The mock provider echoes the question. A real model would answer it.")

    step(3, "Ask something outside the knowledge base")
    question = "What is the capital of France?"
    status, answer = server.call("/ask", {"question": question})
    check(status == 200 and not answer["found"], "Expected a not found result")
    show("question", question)
    show("answer", answer["answer"])
    show("model call", "None. With no relevant text, the model is not called at all.")

    step(4, "The agent looks up an order with a tool")
    status, chat = server.call("/chat", {"message": "What is the status of ORD-1002?"})
    check(status == 200 and chat["tool_calls"][0]["name"] == "lookup_record", "Expected a lookup")
    check("processing" in chat["reply"], "Expected the order to be processing")
    show("tool call", chat["tool_calls"][0])
    show("reply", chat["reply"])

    step(5, "Ask for a change: it only creates a pending approval")
    status, chat = server.call("/chat", {"message": "Please update ORD-1002 to shipped"})
    check(status == 200 and chat["pending_approvals"], "Expected a pending approval")
    approval_id = chat["pending_approvals"][0]
    call = chat["tool_calls"][0]
    check(call["status"] == "pending_approval", "Expected the write to wait for approval")
    show("tool call", {"name": call["name"], "status": call["status"]})
    status, record = server.call("/records/ORD-1002")
    check(record["status"] == "processing", "The record must not change before approval")
    show("record", record)

    step(6, "A person approves it: only now does the record change")
    status, approval = server.call(f"/approvals/{approval_id}", {"decision": "approve"})
    check(status == 200 and approval["status"] == "executed", "Expected the action to run")
    show("approval", {"status": approval["status"], "result": approval["result"]})
    status, record = server.call("/records/ORD-1002")
    check(record["status"] == "shipped", "Expected the record to change after approval")
    show("record", record)

    step(7, "The ticket workflow drafts a reply and waits for approval")
    ticket = {"ticket_id": "T-100", "message": "Where is my delivery for ORD-1001?"}
    status, state = server.call("/workflows/ticket", ticket)
    check(status == 200 and state["status"] == "pending_approval", "Expected a pending approval")
    show("ticket", ticket["message"])
    show("category", state["category"])
    show("order", state["record_id"])
    show("status", state["status"])
    status, approval = server.call(f"/approvals/{state['approval_id']}", {"decision": "approve"})
    check(status == 200 and approval["status"] == "executed", "Expected the reply to be sent")
    status, state = server.call("/workflows/ticket", ticket)
    check(state["status"] == "sent", "Expected the ticket to be sent")
    show("approved", f"status is now {state['status']} (through the stub, so nothing left the app)")

    step(8, "Every tool call is in the audit log")
    entries = server.tool_log_lines()
    check(len(entries) >= 2, "Expected tool calls in the log")
    for entry in entries:
        show("log", f"{entry['tool']} {json.dumps(entry['tool_input'])} -> {entry['outcome']}")

    step(9, "The evaluation runs on a fixed question set")
    result = subprocess.run(
        [sys.executable, "-m", "ai_ops.evaluation", "--label", "demo", "--output-dir", directory],
        cwd=REPO,
        env=server.env,
        capture_output=True,
        text=True,
    )
    check(result.returncode == 0, "Expected the evaluation to pass")
    for line in result.stdout.splitlines():
        if line.startswith(
            ("  hit_rate", "  context_fact", "  not_found", "  injection", "PASSED")
        ):
            print(f"  {line.strip()}")
    show("note", "Mock provider, synthetic data. See docs/EVALUATION.md for the limits.")


def main() -> int:
    """Run the demo and report the result.

    Why: A failed check prints one clear line, with no traceback.

    Used by: the `__main__` block below.

    Returns:
        0 if every check passed, 1 if not.

    Side effects:
        Starts and stops a local server, and writes to a temporary directory.
    """
    # `ignore_cleanup_errors` because Windows can briefly keep a file locked after a process ends.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        directory = Path(temporary)
        try:
            with Server(directory) as server:
                run_steps(server, directory)
        except DemoError as error:
            print(f"\nDemo failed: {error}")
            return 1
    print(f"\n{LINE}\nDemo finished: every check passed.\n{LINE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

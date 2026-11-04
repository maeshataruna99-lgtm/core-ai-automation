"""Load versioned prompt templates from files.

Why: Prompts change model behaviour as much as code does. Keeping them in versioned files
makes changes visible in Git diffs, and lets an evaluation compare two versions.

Used by: services, from Milestone 3 onwards, and `tests/test_llm_mock_and_prompts.py`.
"""

from dataclasses import dataclass
from importlib.resources import files
from string import Template


class PromptNotFoundError(Exception):
    """The requested prompt file does not exist.

    Why: A clear name and version are easier to act on than a raw file path error.

    Used by: `load_prompt`.
    """


@dataclass(frozen=True)
class PromptTemplate:
    """A prompt loaded from a file, ready to be filled in.

    Why: Keeping the name and version with the text lets evaluation reports record exactly
    which prompt produced a result.

    Used by: returned by `load_prompt`.
    """

    name: str
    version: int
    text: str

    def render(self, **values: str) -> str:
        """Fill the `$name` placeholders with values.

        Why: `string.Template` makes one pass, so `$context` inside a document stays plain
        text. `str.format` would also fail on the `{` characters common in documents.

        Used by: services, when they build the messages for a model.

        Args:
            **values: The value for each placeholder.

        Returns:
            The final prompt text.

        Raises:
            KeyError: A placeholder has no value.
        """
        return Template(self.text).substitute(values)


def load_prompt(name: str, version: int) -> PromptTemplate:
    """Read `llm/prompts/<name>.v<version>.txt`.

    Why: `importlib.resources` finds the file relative to the package, so it works from any
    working directory. The version is required, so nobody picks up a new prompt by accident.

    Used by: services. Names come from code, so they are not validated as untrusted input.

    Args:
        name: The prompt name.
        version: The prompt version.

    Returns:
        The loaded template.

    Raises:
        PromptNotFoundError: The file does not exist.
    """
    path = files("ai_ops.llm") / "prompts" / f"{name}.v{version}.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PromptNotFoundError(f"Prompt {name} version {version} does not exist") from None
    return PromptTemplate(name=name, version=version, text=text)

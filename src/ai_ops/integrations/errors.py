"""The base error for integrations.

Why: When an approved action fails, the person who approved it should learn why. Most exception text
is not safe to show, so this type marks an error as written on purpose for a client to read.

Used by: `ai_ops.integrations.records`, `ai_ops.integrations.outbound`, and
`ai_ops.services.approval`, which shows these messages and hides all others.
"""


class IntegrationError(Exception):
    """A failure whose message is safe to show to a client and write to the logs.

    Why: See the module docstring. A message must never hold a key, personal data, or a raw response
    from another service.

    Used by: the subclasses in the integrations package.
    """

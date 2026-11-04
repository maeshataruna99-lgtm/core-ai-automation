from importlib.metadata import version

import ai_ops


def test_package_is_installed_from_src():
    assert version("ai-ops")
    assert ai_ops.__doc__

"""The web interface: plain HTML, CSS, and JavaScript.

Why: This file makes the folder a package, so that its files are included when the project is
installed. The interface has no build step and no dependencies, and only calls the public API.

Used by: `ai_ops.api.app.create_app`, which serves this folder at `/ui/`.
"""

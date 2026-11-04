# OOP Design Notes

This document records actual design decisions as implementation progresses.

## Principles

- Python supports OOP directly; use classes and interfaces where they clarify a real boundary.
- Use `Protocol` or ABC for replaceable components (LLM provider, vector store, repository, tool).
- Use dataclasses or Pydantic models for data; do not add behaviour-free classes with logic.
- Prefer composition and dependency injection over inheritance.
- Keep pure logic (chunking, scoring, prompt building) in functions.
- Prefer simple designs over patterns added for appearance.

## Implementation record

| Concept | Language | File / symbol | Status | Rationale |
|---|---|---|---|---|
| Interface (provider) | Python | Fill after implementation | Planned | Swap mock and real LLM without changing services |
| Interface (vector store) | Python | Fill after implementation | Planned | Move from in-memory to a vector database |
| Repository | Python | Fill after implementation | Planned | Keep SQL out of services and tools |
| Adapter | Python | Fill after implementation | Planned | Isolate the third-party service |
| Tool registry | Python | Fill after implementation | Planned | Typed tools for the agent |
| State machine (workflow/approval) | Python | Fill after implementation | Planned | Explicit pending, approved, rejected states |
| Dependency injection | Python | Fill after implementation | Planned | Compose services in the API layer |

Update this table only with real source paths and symbols after implementation. Do not claim planned patterns are implemented.

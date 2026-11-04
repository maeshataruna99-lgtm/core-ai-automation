# Portfolio Positioning

## Honest project description

> Built an educational AI operations assistant in Python with FastAPI. Implemented an LLM provider layer, Retrieval-Augmented Generation with source citations, a tool-using agent with human approval for write actions, a ticket workflow, SQL integration, and an evaluation suite that tracks retrieval quality, latency, and cost.

Use this wording only after the listed features are implemented and verified.

## Skills demonstrated

- Python application development and API design.
- LLM and Generative AI integration.
- RAG, embeddings, and vector search.
- AI agents, chatbots, and workflow automation.
- Prompt engineering and prompt-injection awareness.
- Database and third-party service integration.
- Evaluation, testing, and optimisation of AI features.
- Documentation and troubleshooting.

## Mapping to the job posting

Responsibilities and requirements in the AI Engineer listing map to the requirements in `docs/PROJECT_SPECIFICATION.md` section 4. Quote only what the repository can show.

## Important limitations to disclose

- Educational project with synthetic data only.
- No production deployment and no real customer data.
- No claim of employer or client experience.
- Accuracy and latency numbers apply only to the project's own evaluation set.
- The optional frontend is a demo, not a production interface.

## Suggested demo sequence

1. Start the API with the mock provider and show the OpenAPI page.
2. Ask a question and show the answer with its sources.
3. Ask something outside the knowledge base and show the "not found" answer.
4. Ask about an order or ticket and show the tool lookup in the log.
5. Trigger a write action and show it waiting for approval.
6. Approve it and show the record change.
7. Run the evaluation command and show the report.

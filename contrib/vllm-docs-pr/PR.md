# Draft pull request against vllm-project/vllm

**NOT FILED.** This is prepared for review. Filing it publishes to a third-party
repository under the account of whoever submits it, so that step is deliberately
left to a human.

## Title

`[Doc] Add a logging configuration page`

## Body

vLLM has five environment variables that control logging, and no documentation
page covering them. `docs/configuration/env_vars.md` renders `vllm/envs.py`
through an mkdocs snippet, so the variables appear in the generated table but
without guidance on when to use which, and the two that interact are not
documented as interacting.

This adds `docs/configuration/logging.md` covering:

- `VLLM_LOGGING_LEVEL` for the log level
- `VLLM_CONFIGURE_LOGGING` for embedding vLLM in an application that owns its
  own logging setup
- `VLLM_LOGGING_CONFIG_PATH` for supplying a `dictConfig` JSON file, including
  that it implies `VLLM_CONFIGURE_LOGGING` and errors when the two conflict
- `VLLM_LOGGING_STREAM` and `VLLM_LOGGING_PREFIX` for redirecting and labelling
  output

Every statement is taken from `vllm/envs.py` and `vllm/logger.py` at the commit
below, not from memory.

Closes the documentation gap behind #6660 ("How to disable logging"), which asks
exactly this and is answered today only by reading the source.

### How this was found

While building an evaluation harness for retrieval over vLLM's documentation, I
assembled a golden set of questions taken from real `usage`-labelled issues and
measured which of them the documentation can actually answer. "How to disable
logging" was one of two questions in the set that no retrieval configuration
could answer, because the answer exists only in `vllm/envs.py`.

## Files

- `docs/configuration/logging.md` (new, in this directory as `logging.md`)
- `docs/configuration/README.md`: add a bullet linking the new page

### Suggested README.md change

Under the three configuration levels, add:

```markdown
- [Logging configuration](./logging.md)
```

## Verification checklist before filing

- [ ] Re-read `vllm/envs.py` and `vllm/logger.py` at current `main`; these
      statements were verified at commit `65f3fca5`
- [ ] Confirm the page renders in a local `mkdocs serve`
- [ ] Confirm #6660 is still the best issue to reference
- [ ] Read CONTRIBUTING.md for the current DCO and title conventions

# Logging Configuration

vLLM configures Python logging on import. This page covers how to change the log
level, redirect or prefix log output, replace the logging configuration
entirely, or turn vLLM's logging configuration off.

All of these are environment variables and must be set before vLLM is imported.

## Changing the log level

Set `VLLM_LOGGING_LEVEL` to any standard Python logging level. The default is
`INFO`.

```bash
VLLM_LOGGING_LEVEL=WARNING vllm serve meta-llama/Llama-3.1-8B
```

The value is upper-cased, so `debug` and `DEBUG` behave the same.

## Turning vLLM's logging configuration off

Set `VLLM_CONFIGURE_LOGGING=0` to stop vLLM configuring logging handlers at all.
vLLM still emits log records; they are handled by whatever logging configuration
your application has already installed.

```bash
VLLM_CONFIGURE_LOGGING=0 vllm serve meta-llama/Llama-3.1-8B
```

This is the option to use when vLLM is embedded in an application that owns its
own logging setup, and vLLM's handlers would otherwise duplicate or override it.

!!! note
    `VLLM_CONFIGURE_LOGGING=0` disables vLLM's *configuration* of logging, not
    the log records themselves. To reduce output, set `VLLM_LOGGING_LEVEL`
    instead.

## Using your own logging configuration

Set `VLLM_LOGGING_CONFIG_PATH` to the path of a JSON file in Python's
[`logging.config.dictConfig`](https://docs.python.org/3/library/logging.config.html#logging-config-dictschema)
format. vLLM loads it in place of its default configuration.

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging.json vllm serve meta-llama/Llama-3.1-8B
```

`VLLM_LOGGING_CONFIG_PATH` implies `VLLM_CONFIGURE_LOGGING`. Setting the path
while `VLLM_CONFIGURE_LOGGING=0` raises an error rather than silently ignoring
one of the two.

## Redirecting and prefixing log output

- `VLLM_LOGGING_STREAM` sets the stream the default handler writes to. It takes
  a `logging`-style external object reference and defaults to
  `ext://sys.stdout`. Use `ext://sys.stderr` to send logs to standard error.
- `VLLM_LOGGING_PREFIX` prepends a fixed string to every log message. This is
  useful when several vLLM processes share a console, for example in
  multi-node or data-parallel deployments.

```bash
VLLM_LOGGING_STREAM=ext://sys.stderr \
VLLM_LOGGING_PREFIX="[rank0] " \
  vllm serve meta-llama/Llama-3.1-8B
```

## Summary

| Variable | Default | Effect |
|---|---|---|
| `VLLM_LOGGING_LEVEL` | `INFO` | Default log level |
| `VLLM_CONFIGURE_LOGGING` | `1` | Whether vLLM configures logging handlers |
| `VLLM_LOGGING_CONFIG_PATH` | unset | Path to a `dictConfig` JSON file |
| `VLLM_LOGGING_STREAM` | `ext://sys.stdout` | Stream the default handler writes to |
| `VLLM_LOGGING_PREFIX` | empty | String prepended to every log message |

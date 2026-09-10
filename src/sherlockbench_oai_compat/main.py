from datetime import datetime
from functools import partial
from pprint import pprint

from openai import (OpenAI, APITimeoutError, InternalServerError,
                    BadRequestError, RateLimitError, APIConnectionError)

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, print_progress_with_estimate
from sherlockbench_client import run_with_error_handling, set_current_attempt

# sampling params the openai sdk accepts as named arguments
NATIVE_SAMPLING = {"temperature", "top_p"}

from .investigate_decide_verify import investigate_decide_verify
from .investigate_verify import investigate_verify
from .prompts import make_initial_messages

def make_completionfn(config, eventlogger):
    """Any server speaking the OpenAI /v1/responses dialect: openrouter,
    llama-server, ollama, and friends. base-url and api-key-name come from the
    model's config entry, since entries in this provider point at different
    servers."""

    # n.b. 'base-url' is already the sherlockbench api server, so the llm
    # endpoint gets its own key
    client = OpenAI(api_key=config['api-keys'][config['api-key-name']],
                    base_url=config['llm-base-url'],
                    timeout=900.0)

    def create_completion(client, **kwargs):
        """closure to pre-load the model"""

        return client.responses.parse(
            **kwargs
        )

    def completionfn(**kwargs):
        # the sdk has named params for these two; everything else rides in
        # extra_body, which is how openrouter takes top_k, min_p and penalties
        for key, value in config.get("sampling", {}).items():
            if key in NATIVE_SAMPLING:
                kwargs[key] = value
            else:
                kwargs.setdefault("extra_body", {})[key] = value

        # no reasoning summaries and no service_tier: both are openai-only
        if "reasoning_effort" in config:
            kwargs["reasoning"] = {"effort": config['reasoning_effort']}

        # these servers disagree about where the schema goes: openrouter honours
        # text.format (what the sdk sends) and ignores response_format;
        # llama-server does the exact opposite. sending both satisfies either.
        if "text_format" in kwargs:
            schema = kwargs["text_format"].model_json_schema()
            schema["additionalProperties"] = False

            kwargs["extra_body"] = {**kwargs.get("extra_body", {}),
                                    "response_format": {
                                        "type": "json_schema",
                                        "json_schema": {"name": kwargs["text_format"].__name__,
                                                        "strict": True,
                                                        "schema": schema}}}

        return create_completion(client, model=config['model'], **kwargs)

    completionfn = LLMRateLimiter(eventlogger, rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=[(APITimeoutError, 300),
                                                      (InternalServerError, 60),
                                                      (RateLimitError, 60),
                                                      (APIConnectionError, 60),
                                                      (BadRequestError, 60)])

    return completionfn

def run_benchmark(executor, config, db_conn, cursor, eventlogger, run_id, attempts, start_time):
    """
    Run the OAI-compatible benchmark with the given parameters.
    This function is called by run_with_error_handling.
    """

    postfn = lambda *args: post(config["base-url"], run_id, *args)

    completionfn = make_completionfn(config, eventlogger)

    executor_p = partial(executor, postfn, completionfn, eventlogger, config, run_id, cursor, make_completionfn)

    for i, attempt in enumerate(attempts, 1):
        print_progress_with_estimate(i, len(attempts), start_time)

        # Track the current attempt for error handling
        set_current_attempt(attempt)

        # Process the attempt
        executor_p(attempt)

        # Clear the current attempt since we've completed processing it
        set_current_attempt(None)

    # Return the values needed for run completion
    return postfn, completionfn.total_call_count, config

def two_phase():
    run_with_error_handling("oai-compat", run_benchmark, investigate_verify)

def three_phase():
    run_with_error_handling("oai-compat", run_benchmark, partial(investigate_decide_verify, False))

def inv_isolated():
    run_with_error_handling("oai-compat", run_benchmark, partial(investigate_decide_verify, "inv_isolated"))

def random_inv():
    run_with_error_handling("oai-compat", run_benchmark, partial(investigate_decide_verify, "random_inv"))

def main():
    run_with_error_handling("oai-compat", run_benchmark, {"2-phase": investigate_verify,
                                                      "3-phase": partial(investigate_decide_verify, False)})

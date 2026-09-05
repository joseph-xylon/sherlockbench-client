from datetime import datetime
from functools import partial

from anthropic import (Anthropic, APITimeoutError, APIConnectionError,
                       InternalServerError, OverloadedError, RateLimitError)

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, print_progress_with_estimate
from sherlockbench_client import run_with_error_handling, set_current_attempt

from .investigate_decide_verify import investigate_decide_verify
from .investigate_verify import investigate_verify
from .prompts import system_prompt

def make_completionfn(config, eventlogger):
    client = Anthropic(api_key=config['api-keys']['anthropic'],
                       timeout=1200.0)

    def create_completion(client, **kwargs):
        """closure to pre-load the model"""

        return client.messages.parse(
            **kwargs
        )

    def completionfn(**kwargs):
        kwargs.setdefault("max_tokens", config.get("max_tokens", 16000))
        kwargs.setdefault("system", system_prompt)

        # adaptive is the only on-mode on these models. budget_tokens is a 400.
        kwargs.setdefault("thinking", {"type": "adaptive", "display": "summarized"})

        if "effort" in config:
            kwargs["output_config"] = {**kwargs.get("output_config", {}),
                                       "effort": config['effort']}

        return create_completion(client, model=config['model'], **kwargs)

    completionfn = LLMRateLimiter(eventlogger, rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=[(OverloadedError, 600),
                                                      (RateLimitError, 60),
                                                      (APITimeoutError, 300),
                                                      (APIConnectionError, 60),
                                                      (InternalServerError, 60)])

    return completionfn

def run_benchmark(executor, config, db_conn, cursor, eventlogger, run_id, attempts, start_time):
    """
    Run the Anthropic benchmark with the given parameters.
    This function is called by run_with_error_handling.
    """
    postfn = lambda *args: post(config["base-url"], run_id, *args)

    completionfn = make_completionfn(config, eventlogger)

    executor_p = partial(executor, postfn, completionfn, eventlogger, config, run_id, cursor)

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
    run_with_error_handling("anthropic", run_benchmark, investigate_verify)

def three_phase():
    run_with_error_handling("anthropic", run_benchmark, partial(investigate_decide_verify, False))

def inv_isolated():
    run_with_error_handling("anthropic", run_benchmark, partial(investigate_decide_verify, "inv_isolated"))

def random_inv():
    run_with_error_handling("anthropic", run_benchmark, partial(investigate_decide_verify, "random_inv"))

def main():
    run_with_error_handling("anthropic", run_benchmark, {"2-phase": investigate_verify,
                                                         "3-phase": partial(investigate_decide_verify, False)})

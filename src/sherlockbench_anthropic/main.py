from datetime import datetime
from functools import partial

import anthropic

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, print_progress_with_estimate
from sherlockbench_client import run_with_error_handling, set_current_attempt

from .investigate_decide_verify import investigate_decide_verify
from .investigate_verify import investigate_verify
from .prompts import make_initial_message

def create_completion(client, model, **kwargs):
    """closure to pre-load the model"""

    thinkingsuffix="+thinking"
    if model.endswith(thinkingsuffix):
        return client.with_options(timeout=1200).messages.create(
            model=model.removesuffix(thinkingsuffix),
            max_tokens=32000,
            thinking={
                "type": "enabled",
                "budget_tokens": 20000
            },
            extra_headers={
                "anthropic-beta": "interleaved-thinking-2025-05-14"
            },
            **kwargs
        )

    else:
        return client.messages.create(
            model=model,
            max_tokens=8192,
            **kwargs
        )

def run_benchmark(executor, config, db_conn, cursor, eventlogger, run_id, attempts, start_time):
    """
    Run the Anthropic benchmark with the given parameters.
    This function is called by run_with_error_handling.
    """
    client = anthropic.Anthropic(api_key=config['api-keys']['anthropic'])

    postfn = lambda *args: post(config["base-url"], run_id, *args)

    def completionfn(**kwargs):
        if "temperature" in config:
            kwargs["temperature"] = config['temperature']

        return create_completion(client, config['model'], **kwargs)

    completionfn = LLMRateLimiter(eventlogger, rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=[(anthropic._exceptions.OverloadedError, 600)])

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
    run_with_error_handling("anthropic", run_benchmark, investigate_decide_verify)

def main():
    run_with_error_handling("anthropic", run_benchmark, {"2-phase": investigate_verify,
                                                         "3-phase": investigate_decide_verify})

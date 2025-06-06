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

def run_benchmark(executor, config, db_conn, cursor, run_id, attempts, start_time):
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

    completionfn = LLMRateLimiter(rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=[(anthropic._exceptions.OverloadedError, 600)])

    assert len(attempts) == 200

    executor_p = partial(executor, postfn, completionfn, config, run_id, cursor)
    chunk_size = 20
    for i in range(0, 200, chunk_size):
        chunk = attempts[i:i + chunk_size]

        executor_p(chunk, i)

    # Return the values needed for run completion
    return postfn, completionfn.total_call_count, config

def two_phase():
    run_with_error_handling("anthropic", run_benchmark, investigate_verify)

def three_phase():
    run_with_error_handling("anthropic", run_benchmark, investigate_decide_verify)

def main():
    run_with_error_handling("anthropic", run_benchmark, {"2-phase": investigate_verify,
                                                         "3-phase": investigate_decide_verify})

from datetime import datetime
from functools import partial

from openai import OpenAI, RateLimitError, APITimeoutError

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, print_progress_with_estimate
from sherlockbench_client import run_with_error_handling, set_current_attempt

from .investigate_decide_verify import investigate_decide_verify
from .investigate_verify import investigate_verify
from .prompts import make_initial_messages

def create_completion(client, **kwargs):
    """closure to pre-load the model"""
    return client.chat.completions.create(
        **kwargs
    )

def run_benchmark(executor, config, db_conn, cursor, eventlogger, run_id, attempts, start_time):
    """
    Run the XAI benchmark with the given parameters.
    This function is called by run_with_error_handling.
    """
    client = OpenAI(base_url="https://api.x.ai/v1",
                    api_key=config['api-keys']['xai'],
                    timeout=900.0)

    postfn = lambda *args: post(config["base-url"], run_id, *args)

    def completionfn(**kwargs):
        if "temperature" in config:
            kwargs["temperature"] = config['temperature']

        if "reasoning_effort" in config:
            kwargs["reasoning_effort"] = config['reasoning_effort']

        return create_completion(client, model=config['model'], **kwargs)

    completionfn = LLMRateLimiter(eventlogger, rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=[(RateLimitError, 300),
                                                      (APITimeoutError, 300)])

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
    run_with_error_handling("xai", run_benchmark, investigate_verify)

def three_phase():
    run_with_error_handling("xai", run_benchmark, partial(investigate_decide_verify, False))

def inv_isolated():
    run_with_error_handling("xai", run_benchmark, partial(investigate_decide_verify, True))

def main():
    run_with_error_handling("xai", run_benchmark, {"2-phase": investigate_verify,
                                                   "3-phase": partial(investigate_decide_verify, False)})

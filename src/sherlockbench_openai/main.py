from datetime import datetime
from functools import partial
from pprint import pprint

from openai import OpenAI, APITimeoutError, InternalServerError

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, print_progress_with_estimate
from sherlockbench_client import run_with_error_handling, set_current_attempt

from .investigate_decide_verify import investigate_decide_verify
from .investigate_verify import investigate_verify
from .prompts import make_initial_messages

def create_completion(client, **kwargs):
    """closure to pre-load the model"""

    return client.beta.chat.completions.parse(
        **kwargs
    )

def run_benchmark(executor, config, db_conn, cursor, run_id, attempts, start_time):
    """
    Run the OpenAI benchmark with the given parameters.
    This function is called by run_with_error_handling.
    """
    client = OpenAI(api_key=config['api-keys']['openai'],
                    timeout=900.0)

    postfn = lambda *args: post(config["base-url"], run_id, *args)

    def completionfn(**kwargs):
        if "temperature" in config:
            kwargs["temperature"] = config['temperature']

        if "reasoning_effort" in config:
            kwargs["reasoning_effort"] = config['reasoning_effort']

        if "service_tier" in config:
            kwargs["service_tier"] = config['service_tier']

        return create_completion(client, model=config['model'], **kwargs)

    completionfn = LLMRateLimiter(rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=[(APITimeoutError, 300),
                                                      (InternalServerError, 60)])

    assert len(attempts) == 200

    executor_p = partial(executor, postfn, completionfn, config, run_id, cursor)
    chunk_size = 20
    for i in range(0, 200, chunk_size):
        chunk = attempts[i:i + chunk_size]

        executor_p(chunk, i)

    # Return the values needed for run completion
    return postfn, completionfn.total_call_count, config

def two_phase():
    run_with_error_handling("openai", run_benchmark, investigate_verify)

def three_phase():
    run_with_error_handling("openai", run_benchmark, investigate_decide_verify)

def main():
    run_with_error_handling("openai", run_benchmark, {"2-phase": investigate_verify,
                                                      "3-phase": investigate_decide_verify})

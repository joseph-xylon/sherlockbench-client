from openai import OpenAI, LengthFinishReasonError, APITimeoutError
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q
from sherlockbench_client import run_with_error_handling, set_current_attempt
from pprint import pprint
from functools import partial

from .prompts import make_initial_messages, make_decision_messages
from .investigate_verify import investigate_verify
from .investigate_decide_verify import investigate_decide_verify
from .verify import verify

from datetime import datetime
import psycopg2

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
    client = OpenAI(api_key=config['api-keys']['openai'])

    postfn = lambda *args: post(config["base-url"], run_id, *args)

    def completionfn(**kwargs):
        if "temperature" in config:
            kwargs["temperature"] = config['temperature']

        if "reasoning_effort" in config:
            kwargs["reasoning_effort"] = config['reasoning_effort']

        return create_completion(client, model=config['model'], **kwargs)

    completionfn = LLMRateLimiter(rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=(APITimeoutError))

    for attempt in attempts:
        # Track the current attempt for error handling
        set_current_attempt(attempt)

        # Process the attempt
        executor(postfn, completionfn, config, attempt, run_id, cursor)

        # Clear the current attempt since we've completed processing it
        set_current_attempt(None)

    # Return the values needed for run completion
    return postfn, completionfn.total_call_count, config

def two_phase():
    # Use the centralized error handling function
    run_with_error_handling("openai", partial(run_benchmark, investigate_verify))

def three_phase():
    # Use the centralized error handling function
    run_with_error_handling("openai", partial(run_benchmark, investigate_decide_verify))

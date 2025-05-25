from openai import OpenAI, LengthFinishReasonError
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q
from sherlockbench_client import run_with_error_handling, set_current_attempt

from .prompts import make_initial_messages
from .investigate_verify import investigate_verify
from .investigate_decide_verify import investigate_decide_verify
from .verify import verify
from functools import partial

from datetime import datetime
import psycopg2

def create_completion(client, **kwargs):
    """closure to pre-load the model"""
    return client.chat.completions.create(
        **kwargs
    )

def run_benchmark(executor, config, db_conn, cursor, run_id, attempts, start_time):
    """
    Run the XAI benchmark with the given parameters.
    This function is called by run_with_error_handling.
    """
    client = OpenAI(base_url="https://api.x.ai/v1",
                    api_key=config['api-keys']['xai'])

    postfn = lambda *args: post(config["base-url"], run_id, *args)

    def completionfn(**kwargs):
        if "temperature" in config:
            kwargs["temperature"] = config['temperature']

        if "reasoning_effort" in config:
            kwargs["reasoning_effort"] = config['reasoning_effort']
            
        return create_completion(client, model=config['model'], **kwargs)

    completionfn = LLMRateLimiter(rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=())

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
    run_with_error_handling("xai", partial(run_benchmark, investigate_verify))

def three_phase():
    # Use the centralized error handling function
    run_with_error_handling("xai", partial(run_benchmark, investigate_decide_verify))

if __name__ == "__main__":
    main()

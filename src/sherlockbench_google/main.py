from datetime import datetime
from functools import partial
from pprint import pprint

from google import genai
from google.genai import types, errors

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, print_progress_with_estimate
from sherlockbench_client import run_with_error_handling, set_current_attempt

from .investigate_decide_verify import investigate_decide_verify
from .investigate_verify import investigate_verify
from .prompts import system_message, make_initial_message
from .utility import save_message

def create_completion(client, tools=None, schema=None, temperature=None, **kwargs):
    """closure to pre-load the model"""
    #print("CONTENTS")
    #print(contents)

    config_args = {
        "system_instruction": system_message,
        #"max_output_tokens": 3
    }

    if temperature is not None:
        config_args["temperature"] = temperature

    if tools is not None:
        config_args["tools"] = tools

    if schema is not None:
        config_args["response_schema"] = schema
        config_args["response_mime_type"] = 'application/json'

    return client.models.generate_content(
        config=types.GenerateContentConfig(**config_args),
        **kwargs
    )

def run_benchmark(executor, config, db_conn, cursor, run_id, attempts, start_time):
    """
    Run the Google benchmark with the given parameters.
    This function is called by run_with_error_handling.
    """
    client = genai.Client(api_key=config['api-keys']['google'])

    postfn = lambda *args: post(config["base-url"], run_id, *args)

    def completionfn(**kwargs):
        if "temperature" in config:
            kwargs["temperature"] = config['temperature']

        return create_completion(client, model=config['model'], **kwargs)

    completionfn = LLMRateLimiter(rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=[(errors.ServerError, 300), (errors.ClientError, 900)])

    assert len(attempts) == 200

    executor_p = partial(executor, postfn, completionfn, config, run_id, cursor)
    chunk_size = 20
    for i in range(0, 200, chunk_size):
        chunk = attempts[i:i + chunk_size]

        executor_p(chunk, i)

    # Return the values needed for run completion
    return postfn, completionfn.total_call_count, config

def two_phase():
    run_with_error_handling("google", run_benchmark, investigate_verify)

def three_phase():
    run_with_error_handling("google", run_benchmark, investigate_decide_verify)

def main():
    run_with_error_handling("google", run_benchmark, {"2-phase": investigate_verify,
                                                      "3-phase": investigate_decide_verify})

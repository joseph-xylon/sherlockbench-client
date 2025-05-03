import sys
from google import genai
from google.genai import types, errors
from pprint import pprint

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q
from sherlockbench_client import run_with_error_handling, set_current_attempt

from .prompts import system_message, initial_message
from .utility import save_message
from .investigate import investigate
from .verify import verify

from datetime import datetime

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

def investigate_and_verify(postfn, completionfn, config, attempt_id, arg_spec, run_id, cursor):
    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = [save_message("user", initial_message)]
    messages, tool_call_count = investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify(config, postfn, completionfn, messages, printer, attempt_id)

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id)

    return verification_result

def run_benchmark(config, db_conn, cursor, run_id, attempts, start_time):
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
                                  backoff_exceptions=(errors.ServerError))
    
    for attempt in attempts:
        # Track the current attempt for error handling
        set_current_attempt(attempt)
        
        # Process the attempt
        investigate_and_verify(postfn, completionfn, config, attempt["attempt-id"], attempt["arg-spec"], run_id, cursor)
        
        # Clear the current attempt since we've completed processing it
        set_current_attempt(None)

    # Return the values needed for run completion
    return postfn, completionfn.total_call_count, config

def main():
    # Use the centralized error handling function
    run_with_error_handling("google", run_benchmark)

if __name__ == "__main__":
    main()

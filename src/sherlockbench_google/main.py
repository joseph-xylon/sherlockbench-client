import sys
from google import genai
from google.genai import types
from pprint import pprint

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, start_run, complete_run

from .prompts import system_message, initial_message
from .utility import save_message
from .investigate import investigate
from .verify import verify

from datetime import datetime

def create_completion(client, model, tools=None, schema=None, **kwargs):
    """closure to pre-load the model"""
    #print("CONTENTS")
    #print(contents)

    config_args = {
        "system_instruction": system_message,
        #"max_output_tokens": 3,
        #"temperature": 0.3,
    }
    
    if tools is not None:
        config_args["tools"] = tools

    if schema is not None:
        config_args["response_schema"] = schema
        config_args["response_mime_type"] = 'application/json'

    return client.models.generate_content(
        model=model,
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

def main():
    config, db_conn, cursor, run_id, attempts, start_time = start_run("google")

    client = genai.Client(api_key=config['api-keys']['google'])

    postfn = lambda *args: post(config["base-url"], run_id, *args)
    completionfn = lambda **kwargs: create_completion(client, config['model'], **kwargs)
    completionfn = LLMRateLimiter(rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=())
    
    for attempt in attempts:
        investigate_and_verify(postfn, completionfn, config, attempt["attempt-id"], attempt["arg-spec"], run_id, cursor)

    complete_run(postfn, db_conn, cursor, run_id, start_time, completionfn.total_call_count, config)

if __name__ == "__main__":
    main()

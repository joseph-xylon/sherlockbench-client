import json
from datetime import datetime
from functools import partial

from pydantic import BaseModel
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, value_list_to_map

from .prompts import make_initial_messages, make_2p_verification_message

from .verify import verify

def list_to_map(input_list):
    """openai doesn't like arrays much so just assign arbritray keys"""
    keys = [chr(97 + i) for i in range(len(input_list))]  # Generate keys: 'a', 'b', 'c', etc.
    return {key: {"type": item} for key, item in zip(keys, input_list)}

def normalize_args(input_dict):
    """Converts a dict into a list of values, sorted by the alphabetical order of the keys."""
    return [input_dict[key] for key in sorted(input_dict.keys())]

def format_inputs(arg_spec, args):
    # Show strings in double-quotes
    fmt_args = list(
        map(
            lambda v, t: f'"{v}"' if t == "string" else v,
            args,
            arg_spec
        )
    )

    if len(fmt_args) > 1:
        return f"({', '.join(map(str, fmt_args))})"
    else:
        return f"{', '.join(map(str, fmt_args))}"

def format_tool_call(args, arg_spec, output_type, result):
    # Clean inputs to handle surrogate characters that can't be encoded
    clean_args = []
    for arg in args:
        if isinstance(arg, str):
            # Replace surrogates with replacement character
            arg = arg.encode('utf-8', 'replace').decode('utf-8')
        clean_args.append(arg)

    # Clean result similarly if it's a string
    if isinstance(result, str):
        result = result.encode('utf-8', 'replace').decode('utf-8')

    if output_type == "string":
        oput = f'"{result}"'
    else:
        oput = result

    return f"{format_inputs(arg_spec, clean_args)} → {oput}"

def handle_tool_call(postfn, printer, attempt_id, arg_spec, output_type, call):
    arguments = json.loads(call.function.arguments)
    args_norm = normalize_args(arguments)

    fnoutput = postfn("test-function", {"attempt-id": attempt_id,
                                        "args": args_norm})["output"]

    printer.indented_print(format_tool_call(args_norm, arg_spec, output_type, fnoutput))

    function_call_result_message = {
        "role": "tool",
        "content": json.dumps(fnoutput),
        "tool_call_id": call.id
    }

    return function_call_result_message

class NoToolException(Exception):
    """When the LLM doesn't use it's tool when it was expected to."""
    pass

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

def investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec, output_type, test_limit):
    mapped_args = list_to_map(arg_spec)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "mystery_function",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": mapped_args,
                    "required": list(mapped_args.keys()),
                    "additionalProperties": False
                },
            },
        }
    ]

    # call the LLM repeatedly until it stops calling it's tool
    tool_call_counter = 0
    for _ in range(0, test_limit + 5):  # the primary limit is on tool calls. This is just a failsafe
        completion = completionfn(messages=messages, tools=tools)

        response = completion.choices[0]
        message = response.message.content
        tool_calls = response.message.tool_calls

        printer.print("\n--- LLM ---")
        printer.indented_print(message)

        if tool_calls:
            printer.print("\n### SYSTEM: calling tool")
            messages.append({"role": "assistant",
                             "content": message,
                             "tool_calls": tool_calls})

            handle_tool_call_p = partial(handle_tool_call, postfn, printer, attempt_id, arg_spec, output_type)
            for call in tool_calls:
                messages.append(handle_tool_call_p(call))

                tool_call_counter += 1

        # if it didn't call the tool we can move on to verifications
        else:
            printer.print("\n### SYSTEM: The tool was used", tool_call_counter, "times.")
            messages.append({"role": "assistant",
                             "content": message})

            return (messages, tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")


def investigate_verify(postfn, completionfn, eventlogger, config, run_id, cursor, attempt):
    attempt_id, arg_spec, output_type, test_limit = destructure(attempt, "attempt-id", "arg-spec", "output-type", "test-limit")

    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = make_initial_messages(test_limit)
    messages, tool_call_count = investigate(config, postfn, completionfn, messages,
                                            printer, attempt_id, arg_spec, output_type, test_limit)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify(config, postfn, completionfn, eventlogger, messages, printer, attempt_id, value_list_to_map, make_2p_verification_message)

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id)

    return verification_result

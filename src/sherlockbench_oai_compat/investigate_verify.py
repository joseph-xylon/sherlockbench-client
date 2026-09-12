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

def make_tools(arg_spec):
    """the tool schema, in the flat shape the responses api expects"""
    mapped_args = list_to_map(arg_spec)

    return [
        {
            "type": "function",
            "name": "mystery_function",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": mapped_args,
                "required": list(mapped_args.keys()),
                "additionalProperties": False
            },
        }
    ]

def handle_tool_call(postfn, printer, attempt_id, arg_spec, output_type, call):
    arguments = json.loads(call.arguments)
    args_norm = normalize_args(arguments)

    fnoutput = postfn("test-function", {"attempt-id": attempt_id,
                                        "args": args_norm})["output"]

    printer.indented_print(format_tool_call(args_norm, arg_spec, output_type, fnoutput))

    function_call_result_message = {
        "type": "function_call_output",
        "output": json.dumps(fnoutput),
        "call_id": call.call_id
    }

    return function_call_result_message

class NoToolException(Exception):
    """When the LLM doesn't use it's tool when it was expected to."""
    pass

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

def print_output(printer, response):
    """print the reasoning summary, if we asked for one, then the message"""
    summaries = [s.text for item in response.output if item.type == "reasoning"
                 for s in item.summary]

    if summaries:
        printer.print("\n--- REASONING ---")
        for summary in summaries:
            printer.indented_print(summary)

    printer.print("\n--- LLM ---")
    printer.indented_print(response.output_text)

def investigate(config, postfn, completionfn, eventlogger, messages, printer, attempt_id, arg_spec, output_type, test_limit):
    tools = make_tools(arg_spec)

    # call the LLM repeatedly until it stops calling it's tool
    tool_call_counter = 0
    for _ in range(0, test_limit + 5):  # the primary limit is on tool calls. This is just a failsafe
        response = completionfn(input=messages, tools=tools,
                                parallel_tool_calls=False)

        tool_calls = [item for item in response.output if item.type == "function_call"]

        print_output(printer, response)

        # append the output so each reasoning item keeps the item it belongs to
        # directly after it -- that is what carries the model's reasoning across
        # tool calls. parallel calls we won't answer are dropped: some providers
        # (Mistral) reject a transcript whose function_call count doesn't match
        # the number of responses.
        answered = {call.call_id for call in tool_calls[:1]}
        messages += [item for item in response.output
                     if item.type != "function_call" or item.call_id in answered]

        if tool_calls:
            printer.print("\n### SYSTEM: calling tool")

            # these servers ignore parallel_tool_calls, so we answer only the
            # first call and drop the rest from the transcript.
            if len(tool_calls) > 1:
                printer.print("### SYSTEM: ignoring", len(tool_calls) - 1, "parallel tool call(s)")
                eventlogger("parallel-tool-calls")

            handle_tool_call_p = partial(handle_tool_call, postfn, printer, attempt_id, arg_spec, output_type)
            messages.append(handle_tool_call_p(tool_calls[0]))

            tool_call_counter += 1

        # if it didn't call the tool we can move on to verifications
        else:
            printer.print("\n### SYSTEM: The tool was used", tool_call_counter, "times.")

            return (messages, tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")


def investigate_verify(postfn, completionfn, eventlogger, config, run_id, cursor, _, attempt):
    attempt_id, arg_spec, output_type, test_limit = destructure(attempt, "attempt-id", "arg-spec", "output-type", "test-limit")

    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = make_initial_messages(test_limit)
    messages, tool_call_count = investigate(config, postfn, completionfn, eventlogger, messages,
                                            printer, attempt_id, arg_spec, output_type, test_limit)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify(config, postfn, completionfn, eventlogger, messages, printer, attempt_id, value_list_to_map, make_2p_verification_message)

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id)

    return verification_result

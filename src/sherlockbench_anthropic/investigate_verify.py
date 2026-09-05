import json
from datetime import datetime
from functools import partial
from pprint import pprint

from sherlockbench_client import destructure, AccumulatingPrinter, q, value_list_to_map

from .prompts import make_initial_message, make_2p_verification_message
from .verify import verify

def list_to_map(input_list):
    """assign arbritray keys to each argument and format it how Anthropic likes"""
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

class NoToolException(Exception):
    """When the LLM doesn't use it's tool when it was expected to."""
    pass

class RefusalException(Exception):
    """When the model declines the request outright."""
    pass

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

def format_tool_call(args, arg_spec, output_type, result):
    if output_type == "string":
        oput = f'"{result}"'
    else:
        oput = result

    return f"{format_inputs(arg_spec, args)} → {oput}"

def make_tools(arg_spec):
    mapped_args = list_to_map(arg_spec)

    return [
        {
            "name": "mystery_function",
            "description": "Use this tool to test the mystery function.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": mapped_args,
                "required": list(mapped_args.keys()),
                "additionalProperties": False
            }
        }
    ]

def print_output(printer, completion):
    """print the thinking summary, if the model gave one, then the text"""
    summaries = [b.thinking for b in completion.content
                 if b.type == "thinking" and b.thinking]

    if summaries:
        printer.print("\n--- REASONING ---")
        for summary in summaries:
            printer.indented_print(summary)

    printer.print("\n--- LLM ---")
    for block in completion.content:
        if block.type == "text":
            printer.indented_print(block.text)

def handle_tool_call(postfn, printer, attempt_id, arg_spec, output_type, call):
    arguments = call.input
    call_id = call.id
    args_norm = normalize_args(arguments)

    response = postfn("test-function", {"attempt-id": attempt_id,
                                        "args": args_norm})

    # Handle case where the output key is missing
    fnoutput = response.get("output", "Error calling tool")
    fnerror = response.get("error", False)

    printer.indented_print(format_tool_call(args_norm, arg_spec, output_type, fnoutput))

    function_call_result_message = {"type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": json.dumps(fnoutput)}

    if fnerror:
        function_call_result_message["is_error"] = True

    return function_call_result_message

def investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec, output_type, test_limit):
    tools = make_tools(arg_spec)

    # call the LLM repeatedly until it stops calling it's tool
    tool_call_counter = 0
    for _ in range(0, test_limit + 5):  # the primary limit is on tool calls. This is just a failsafe
        completion = completionfn(messages=messages, tools=tools,
                                  tool_choice={"type": "auto",
                                               "disable_parallel_tool_use": True})

        if completion.stop_reason == "refusal":
            raise RefusalException(completion.stop_details)

        tool_calls = [b for b in completion.content if b.type == "tool_use"]

        print_output(printer, completion)

        # echo the content back verbatim so the thinking blocks keep their
        # order and signatures. reconstructing it invalidates them.
        if completion.content:
            messages.append({"role": "assistant", "content": completion.content})

        if tool_calls:
            printer.print("\n### SYSTEM: calling tool")

            tool_call_user_message = {
                "role": "user",
                "content": []
            }

            handle_tool_call_p = partial(handle_tool_call, postfn, printer, attempt_id, arg_spec, output_type)
            for call in tool_calls:
                tool_call_user_message["content"].append(handle_tool_call_p(call))

                tool_call_counter += 1

            messages.append(tool_call_user_message)

        # if it didn't call the tool we can move on to verifications
        else:
            printer.print("\n### SYSTEM: The tool was used", tool_call_counter, "times.")

            return (messages, tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")

def investigate_verify(postfn, completionfn, eventlogger, config, run_id, cursor, attempt):
    attempt_id, arg_spec, output_type, test_limit = destructure(attempt, "attempt-id", "arg-spec", "output-type", "test-limit")

    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = make_initial_message(test_limit)
    messages, tool_call_count = investigate(config, postfn, completionfn, messages,
                                            printer, attempt_id, arg_spec, output_type, test_limit)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify(config, postfn, completionfn, eventlogger, messages, printer, attempt_id, value_list_to_map, make_2p_verification_message)

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id)

    return verification_result

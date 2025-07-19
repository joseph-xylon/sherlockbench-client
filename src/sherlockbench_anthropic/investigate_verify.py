import json
from datetime import datetime
from functools import partial
from pprint import pprint

from anthropic.types import TextBlock, ToolUseBlock, ThinkingBlock, RedactedThinkingBlock

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

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

def format_tool_call(args, arg_spec, output_type, result):
    if output_type == "string":
        oput = f'"{result}"'
    else:
        oput = result

    return f"{format_inputs(arg_spec, args)} → {oput}"

def parse_completion(content):
    #text = next((d["text"] for d in content if d.get("type") == "text"), None)
    #tool = next((d["input"] for d in content if d.get("type") == "tool_use"), None)

    # using next allows us to have a default value
    thinking_block = next((item for item in content if isinstance(item, ThinkingBlock)), None)
    redacted_thinking_block = next((item for item in content if isinstance(item, RedactedThinkingBlock)), None)
    text = next((item.text for item in content if isinstance(item, TextBlock)), None)
    tool = [item for item in content if isinstance(item, ToolUseBlock)]

    return (thinking_block, redacted_thinking_block, text, tool)

def handle_tool_call(postfn, printer, attempt_id, arg_spec, output_type, call):
    arguments = call.input
    call_id = call.id
    args_norm = normalize_args(arguments)

    response = postfn("test-function", {"attempt-id": attempt_id,
                                        "args": args_norm})

    # Handle case where the output key is missing
    fnoutput = response.get("output", "Error calling tool")

    printer.indented_print(format_tool_call(args_norm, arg_spec, output_type, fnoutput))

    function_call_result_message = {"type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": json.dumps(fnoutput)}

    return function_call_result_message

def investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec, output_type, test_limit):
    mapped_args = list_to_map(arg_spec)
    tools = [
        {
            "name": "mystery_function",
            "description": "Use this tool to test the mystery function.",
            "input_schema": {
                "type": "object",
                "properties": mapped_args,
                "required": list(mapped_args.keys())
            }
        }
    ]

    # call the LLM repeatedly until it stops calling it's tool
    tool_call_counter = 0
    for _ in range(0, test_limit + 5):  # the primary limit is on tool calls. This is just a failsafe
        #pprint(messages)
        completion = completionfn(messages=messages, tools=tools)

        thinking, redacted_thinking, message, tool_calls = parse_completion(completion.content)

        printer.print("\n--- LLM ---")
        printer.indented_print(message)

        if tool_calls:
            printer.print("\n### SYSTEM: calling tool")
            # Add thinking block for models with +thinking suffix
            content_blocks = []

            if thinking:
                # Convert the ThinkingBlock object to a dict for the API
                content_blocks.append({"type": "thinking", "thinking": thinking.thinking, "signature": thinking.signature})

            if redacted_thinking:
                # Handle redacted thinking block
                content_blocks.append({"type": "redacted_thinking"})

            if message is not None:
                content_blocks.append({"type": "text", "text": message})

            content_blocks.extend(tool_calls)

            messages.append({"role": "assistant", "content": content_blocks})

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

            content_blocks = []

            if thinking:
                # Convert the ThinkingBlock object to a dict for the API
                content_blocks.append({"type": "thinking", "thinking": thinking.thinking, "signature": thinking.signature})

            if redacted_thinking:
                # Handle redacted thinking block
                content_blocks.append({"type": "redacted_thinking"})

            if message is not None:
                content_blocks.append({"type": "text", "text": message})

            messages.append({"role": "assistant", "content": content_blocks})

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

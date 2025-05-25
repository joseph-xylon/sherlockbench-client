import sys
import time
from google.genai import types
from .utility import save_message
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q
from datetime import datetime
from .prompts import system_message, make_initial_message
from .verify import verify

class NoToolException(Exception):
    """When the LLM doesn't use it's tool when it was expected to."""
    pass

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

def generate_schema(input_types):
    # Generate a dictionary with keys as sequential letters and values as types.Schema objects
    schema = {
        chr(97 + i): types.Schema(type=type_str.upper())  # chr(97) is 'a', chr(98) is 'b', etc.
        for i, type_str in enumerate(input_types)
    }
    return schema

def normalize_args(input_dict):
    """Converts a dict into a list of values, sorted by the alphabetical order of the keys."""
    return [input_dict[key] for key in sorted(input_dict.keys())]

def print_tool_call(printer, args, result):
    printer.indented_print(", ".join(map(str, args)), "→", result)

def handle_tool_call(postfn, printer, attempt_id, call):
    arguments = call.args
    fnname = call.name
    args_norm = normalize_args(arguments)

    fnoutput = postfn("test-function", {"attempt-id": attempt_id,
                                        "args": args_norm})["output"]

    print_tool_call(printer, args_norm, fnoutput)

    function_response_content = types.Content(
        role='tool', parts=[types.Part.from_function_response(
            name=fnname,
            response={'result': fnoutput},
        )]
    )

    return function_response_content

def get_text_from_completion(obj_list):
    """
    Concatenates the .text property from each object in the list.
    If an object doesn't have a .text property, it is skipped.
    
    :param obj_list: List of objects to process
    :return: Concatenated string of all .text properties
    """
    result = ""
    for obj in obj_list.candidates[0].content.parts:
        # Use getattr with a default value to avoid AttributeError
        text = getattr(obj, "text", None)
        if text is not None:
            result += text
    return result


def investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec, test_limit):
    mapped_args = generate_schema(arg_spec)
    required_args = list(mapped_args.keys())
    function = types.FunctionDeclaration(
        name='mystery_function',
        description='call this function to investigate what it does',
        parameters=types.Schema(
            type='OBJECT',
            properties=mapped_args,
            required=required_args,
        ),
    )

    tools = [types.Tool(function_declarations=[function])]

    # call the LLM repeatedly until it stops calling it's tool
    tool_call_counter = 0
    for _ in range(0, test_limit + 5):  # the primary limit is on tool calls. This is just a failsafe
        # sometimes gemini-2.5-pro returns None
        attempts = 0
        for _ in range(3):
            completion = completionfn(contents=messages, tools=tools)

            if completion.candidates is None:
                print("Got None response. Retrying after delay.")
                time.sleep(60)
            else:    
                break

        message = get_text_from_completion(completion)
        tool_calls = completion.function_calls

        printer.print("\n--- LLM ---")
        printer.indented_print(message)

        if tool_calls:
            printer.print("\n### SYSTEM: calling tool")
            for part in completion.candidates[0].content.parts:
                messages.append(part)

                if part.function_call is not None:
                    messages.append(handle_tool_call(postfn, printer, attempt_id, part.function_call))
                    tool_call_counter += 1

        # if it didn't call the tool we can move on to verifications
        else:
            printer.print("\n### SYSTEM: The tool was used", tool_call_counter, "times.")
            messages.append(save_message("assistant", message))

            return (messages, tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")

def investigate_verify(postfn, completionfn, config, attempt, run_id, cursor):
    attempt_id, arg_spec, test_limit = destructure(attempt, "attempt-id", "arg-spec", "test-limit")

    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = [save_message("user", make_initial_message(test_limit))]
    messages, tool_call_count = investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec, test_limit)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify(config, postfn, completionfn, messages, printer, attempt_id)

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id)

    return verification_result

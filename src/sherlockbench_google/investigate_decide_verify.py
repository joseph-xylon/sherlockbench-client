import sys
import time
from datetime import datetime
from functools import partial

from google.genai import types
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, set_current_attempt

from .investigate_verify import generate_schema, normalize_args, format_tool_call, format_inputs
from .prompts import system_message, make_initial_message, make_decision_message
from .utility import save_message
from .verify import verify

class NoToolException(Exception):
    """When the LLM doesn't use it's tool when it was expected to."""
    pass

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

class ToolCallHandler:
    def __init__(self, postfn, printer, attempt_id, arg_spec, output_type):
        self.postfn = postfn
        self.printer = printer
        self.attempt_id = attempt_id
        self.arg_spec = arg_spec
        self.output_type = output_type
        self.call_history = []

    def handle_tool_call(self, call):
        arguments = call.args
        fnname = call.name
        args_norm = normalize_args(arguments)

        fnoutput, fnerror = destructure(self.postfn("test-function", {"attempt-id": self.attempt_id,
                                                                       "args": args_norm}),
                                         "output",
                                         "error")

        self.printer.indented_print(format_tool_call(args_norm, self.arg_spec, self.output_type, fnoutput))

        if not fnerror:
            self.call_history.append((args_norm, fnoutput))

        function_response_content = types.Content(
            role='tool', parts=[types.Part.from_function_response(
                name=fnname,
                response={'result': fnoutput},
            )]
        )

        return function_response_content

    def get_call_history(self):
        return self.call_history

    def format_call_history(self):
        lines = []
        for args, output in self.call_history:
            lines.append(format_tool_call(args, self.arg_spec, self.output_type, output))
        return "\n".join(lines)

def get_text_from_completion(obj_list):
    """
    Concatenates the .text property from each object in the list.
    If an object doesn't have a .text property, it is skipped.

    :param obj_list: List of objects to process
    :return: Concatenated string of all .text properties
    """
    # print(f"DEBUG: obj_list type: {type(obj_list)}")
    # print(f"DEBUG: obj_list.candidates: {obj_list.candidates}")
    # print(f"DEBUG: obj_list attributes: {dir(obj_list)}")
    # print(f"DEBUG: obj_list.__dict__: {obj_list.__dict__}")

    if obj_list.candidates is None:
        print("DEBUG: candidates is None")
        raise RuntimeError("API returned None candidates")

    # print(f"DEBUG: obj_list.candidates[0]: {obj_list.candidates[0]}")
    # print(f"DEBUG: obj_list.candidates[0].content: {obj_list.candidates[0].content}")
    if obj_list.candidates[0].content is None:
        print("DEBUG: content is None")
        raise RuntimeError("API returned None content")

    result = ""
    for obj in obj_list.candidates[0].content.parts:
        # Use getattr with a default value to avoid AttributeError
        text = getattr(obj, "text", None)
        if text is not None:
            result += text
    return result


def investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec, output_type, test_limit):
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

    tool_handler = ToolCallHandler(postfn, printer, attempt_id, arg_spec, output_type)

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
                    messages.append(tool_handler.handle_tool_call(part.function_call))
                    tool_call_counter += 1

        # if it didn't call the tool we can move on to verifications
        else:
            printer.print("\n### SYSTEM: The tool was used", tool_call_counter, "times.")
            messages.append(save_message("model", message))

            return (tool_handler.format_call_history(), tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")

def decision(completionfn, messages, printer):
    for _ in range(3):
        completion = completionfn(contents=messages)

        if completion.candidates is None:
            print("Got None response. Retrying after delay.")
            time.sleep(60)
        else:
            break

    message = get_text_from_completion(completion)

    printer.print("\n--- LLM ---")
    printer.indented_print(message)

    messages.append(save_message("model", message))

    return messages

def investigate_decide_verify(postfn, completionfn, config, run_id, cursor, attempts, i):
    attempt_id, arg_spec, output_type, test_limit = destructure(attempts[0], "attempt-id", "arg-spec", "output-type", "test-limit")

    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = [save_message("user", make_initial_message(test_limit))]
    tool_calls, tool_call_count = investigate(config, postfn, completionfn, messages,
                                              printer, attempt_id, arg_spec, output_type, test_limit)

    for idx, attempt in enumerate(attempts, start=1):
        # Track the current attempt for error handling
        set_current_attempt(attempt)

        attempt_id = attempt["attempt-id"]
        start_time = datetime.now()

        printer.print("\n### SYSTEM: making decision based on tool calls", arg_spec)
        printer.print(tool_calls)

        messages = [save_message("user", make_decision_message(tool_calls))]
        messages = decision(completionfn, messages, printer)

        printer.print(f"\n### SYSTEM: verifying function i{i}q{idx}")
        verification_result = verify(config, postfn, completionfn, messages, printer, attempt_id, partial(format_inputs, arg_spec))

        time_taken = (datetime.now() - start_time).total_seconds()
        q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id, {"i": i, "v": idx})

    return verification_result

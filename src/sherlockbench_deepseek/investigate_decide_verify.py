import json
from datetime import datetime
from functools import partial

from pydantic import BaseModel
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, ISOLATED_CONFIG

from .investigate_verify import list_to_map, normalize_args, format_tool_call, format_inputs
from .prompts import make_initial_messages, make_decision_messages, make_3p_verification_message
from .verify import verify

# for inv-isolated mode
from sherlockbench_openai import make_completionfn
from sherlockbench_openai import decision as decision_isolated
from sherlockbench_openai import make_decision_messages as make_decision_messages_isolated
from sherlockbench_openai import make_3p_verification_message as make_3p_verification_message_isolated
from sherlockbench_openai import verify as verify_isolated

class ToolCallHandler:
    def __init__(self, postfn, printer, attempt_id, arg_spec, output_type, eventlogger):
        self.postfn = postfn
        self.printer = printer
        self.attempt_id = attempt_id
        self.arg_spec = arg_spec
        self.output_type = output_type
        self.eventlogger = eventlogger
        self.call_history = []

    def handle_tool_call(self, call):
        try:
            arguments = json.loads(call.function.arguments)

        except json.JSONDecodeError as e:
            function_call_result_message = {
                "role": "tool",
                "content": "invalid json when calling tool",
                "tool_call_id": call.id
            }

            self.eventlogger("invalid-tool-call")

            return function_call_result_message

        args_norm = normalize_args(arguments)

        fnoutput, fnerror = destructure(self.postfn("test-function", {"attempt-id": self.attempt_id,
                                                                      "args": args_norm}),
                                        "output",
                                        "error")

        self.printer.indented_print(format_tool_call(args_norm, self.arg_spec, self.output_type, fnoutput))

        if not fnerror:
            self.call_history.append((args_norm, fnoutput))

        function_call_result_message = {
            "role": "tool",
            "content": json.dumps(fnoutput),
            "tool_call_id": call.id
        }

        return function_call_result_message

    def get_call_history(self):
        return self.call_history

    def format_call_history(self):
        lines = []
        for args, output in self.call_history:
            lines.append(format_tool_call(args, self.arg_spec, self.output_type, output))
        return "\n".join(lines)

class NoToolException(Exception):
    """When the LLM doesn't use it's tool when it was expected to."""
    pass

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

def investigate(config, postfn, completionfn, eventlogger, messages, printer, attempt_id, arg_spec, output_type, test_limit):
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

    tool_handler = ToolCallHandler(postfn, printer, attempt_id, arg_spec, output_type, eventlogger)

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

            for call in tool_calls:
                messages.append(tool_handler.handle_tool_call(call))

                tool_call_counter += 1

        # if it didn't call the tool we can move on to verifications
        else:
            printer.print("\n### SYSTEM: The tool was used", tool_call_counter, "times.")
            messages.append({"role": "assistant",
                             "content": message})

            return (tool_handler.format_call_history(), tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")

def decision(completionfn, messages, printer):
    completion = completionfn(messages=messages)

    response = completion.choices[0]
    message = response.message.content

    printer.print("\n--- LLM ---")
    printer.indented_print(message)

    messages.append({"role": "assistant",
                            "content": message})

    return messages

def investigate_decide_verify(isolated, postfn, completionfn, eventlogger, config, run_id, cursor, attempt):
    attempt_id, arg_spec, output_type, test_limit = destructure(attempt, "attempt-id", "arg-spec", "output-type", "test-limit")

    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = make_initial_messages(test_limit)
    tool_calls, tool_call_count = investigate(config, postfn, completionfn, eventlogger, messages,
                                              printer, attempt_id, arg_spec, output_type, test_limit)

    printer.print("\n### SYSTEM: making decision based on tool calls", arg_spec)
    printer.print(tool_calls)

    if isolated:
        assert ISOLATED_CONFIG is not None, "Error: o4-mini-medium needs to be configured to use this test mode."
        decision_ = decision_isolated
        completionfn_ = make_completionfn(ISOLATED_CONFIG, eventlogger)
        make_decision_messages_ = make_decision_messages_isolated
        make_3p_verification_message_ = make_3p_verification_message_isolated
        verify_ = verify_isolated
    else:
        decision_ = decision
        completionfn_ = completionfn
        make_decision_messages_ = make_decision_messages
        make_3p_verification_message_ = make_3p_verification_message
        verify_ = verify

    messages = make_decision_messages_(tool_calls)
    messages = decision_(completionfn_, messages, printer)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify_(config, postfn, completionfn_, eventlogger, messages, printer, attempt_id, partial(format_inputs, arg_spec), make_3p_verification_message_)

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id)

    return verification_result

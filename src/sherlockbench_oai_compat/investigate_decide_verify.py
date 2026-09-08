import json
from datetime import datetime
from functools import partial

from pydantic import BaseModel
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, ISOLATED_CONFIG

from .investigate_verify import normalize_args, format_tool_call, format_inputs, make_tools, print_output
from .prompts import make_initial_messages, make_decision_messages, make_3p_verification_message
from .verify import verify

class ToolCallHandler:
    def __init__(self, postfn, printer, attempt_id, arg_spec, output_type):
        self.postfn = postfn
        self.printer = printer
        self.attempt_id = attempt_id
        self.arg_spec = arg_spec
        self.output_type = output_type
        self.call_history = []

    def handle_tool_call(self, call):
        arguments = json.loads(call.arguments)
        args_norm = normalize_args(arguments)

        fnoutput, fnerror = destructure(self.postfn("test-function", {"attempt-id": self.attempt_id,
                                                                      "args": args_norm}),
                                        "output",
                                        "error")

        self.printer.indented_print(format_tool_call(args_norm, self.arg_spec, self.output_type, fnoutput))

        if not fnerror:
            self.call_history.append((args_norm, fnoutput))

        function_call_result_message = {
            "type": "function_call_output",
            "output": json.dumps(fnoutput),
            "call_id": call.call_id
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
    tools = make_tools(arg_spec)

    tool_handler = ToolCallHandler(postfn, printer, attempt_id, arg_spec, output_type)

    # call the LLM repeatedly until it stops calling it's tool
    tool_call_counter = 0
    for _ in range(0, test_limit + 5):  # the primary limit is on tool calls. This is just a failsafe
        response = completionfn(input=messages, tools=tools,
                                parallel_tool_calls=False)

        tool_calls = [item for item in response.output if item.type == "function_call"]

        print_output(printer, response)

        messages += response.output

        if tool_calls:
            printer.print("\n### SYSTEM: calling tool")

            # only the first call is answered; see investigate_verify
            if len(tool_calls) > 1:
                printer.print("### SYSTEM: ignoring", len(tool_calls) - 1, "parallel tool call(s)")
                eventlogger("parallel-tool-calls")

            messages.append(tool_handler.handle_tool_call(tool_calls[0]))

            tool_call_counter += 1

        # if it didn't call the tool we can move on to verifications
        else:
            printer.print("\n### SYSTEM: The tool was used", tool_call_counter, "times.")

            return (tool_handler.format_call_history(), tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")

def decision(completionfn, messages, printer):
    response = completionfn(input=messages)

    print_output(printer, response)

    messages += response.output

    return messages

def investigate_decide_verify(experiment, postfn, completionfn, eventlogger, config, run_id, cursor, make_completionfn, attempt):
    attempt_id, arg_spec, output_type, test_limit = destructure(attempt, "attempt-id", "arg-spec", "output-type", "test-limit")

    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    if experiment == "random_inv":
        printer.print("\n### SYSTEM: getting random investigation for args", arg_spec)

        tool_calls = postfn("developer/random-investigation", {"attempt-id": attempt_id})["output"]
        tool_call_count = test_limit

    else:
        printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

        messages = make_initial_messages(test_limit)
        tool_calls, tool_call_count = investigate(config, postfn, completionfn, eventlogger, messages,
                                                  printer, attempt_id, arg_spec, output_type, test_limit)

    printer.print("\n### SYSTEM: making decision based on tool calls", arg_spec)
    printer.print(tool_calls)

    if experiment == "inv_isolated":
        assert ISOLATED_CONFIG is not None, "Error: o4-mini-medium needs to be configured to use this test mode."
        completionfn_ = make_completionfn(ISOLATED_CONFIG, eventlogger)
    else:
        completionfn_ = completionfn

    messages = make_decision_messages(tool_calls)
    messages = decision(completionfn_, messages, printer)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify(config, postfn, completionfn_, eventlogger, messages, printer, attempt_id, partial(format_inputs, arg_spec), make_3p_verification_message)

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id)

    return verification_result

import json
from datetime import datetime
from functools import partial
from pprint import pprint

from anthropic.types import TextBlock, ToolUseBlock, ThinkingBlock, RedactedThinkingBlock
from sherlockbench_client import destructure, AccumulatingPrinter, q, set_current_attempt

from .investigate_verify import list_to_map, normalize_args, format_tool_call, format_inputs, NoToolException, MsgLimitException, parse_completion
from .prompts import make_initial_message, make_decision_messages
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
        arguments = call.input
        call_id = call.id
        args_norm = normalize_args(arguments)

        fnoutput, fnerror = destructure(self.postfn("test-function", {"attempt-id": self.attempt_id,
                                                                      "args": args_norm}),
                                        "output",
                                        "error")

        # Handle case where the output key is missing
        if fnoutput is None:
            fnoutput = "Error calling tool"

        self.printer.indented_print(format_tool_call(args_norm, self.arg_spec, self.output_type, fnoutput))

        if not fnerror:
            self.call_history.append((args_norm, fnoutput))

        function_call_result_message = {"type": "tool_result",
                                        "tool_use_id": call_id,
                                        "content": json.dumps(fnoutput)}

        return function_call_result_message

    def get_call_history(self):
        return self.call_history

    def format_call_history(self):
        lines = []
        for args, output in self.call_history:
            lines.append(format_tool_call(args, self.arg_spec, self.output_type, output))
        return "\n".join(lines)

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

    tool_handler = ToolCallHandler(postfn, printer, attempt_id, arg_spec, output_type)

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

            for call in tool_calls:
                tool_call_user_message["content"].append(tool_handler.handle_tool_call(call))

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

            return (tool_handler.format_call_history(), tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")

def decision(completionfn, messages, printer):
    completion = completionfn(messages=messages)

    thinking, redacted_thinking, message, tool_calls = parse_completion(completion.content)

    printer.print("\n--- LLM ---")
    printer.indented_print(message)

    messages.append({"role": "assistant", "content": message})

    return messages

def investigate_decide_verify(postfn, completionfn, config, run_id, cursor, attempts, i):
    attempt_id, arg_spec, output_type, test_limit = destructure(attempts[0], "attempt-id", "arg-spec", "output-type", "test-limit")

    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = make_initial_message(test_limit)
    tool_calls, tool_call_count = investigate(config, postfn, completionfn, messages,
                                              printer, attempt_id, arg_spec, output_type, test_limit)

    for idx, attempt in enumerate(attempts, start=1):
        # Track the current attempt for error handling
        set_current_attempt(attempt)

        attempt_id = attempt["attempt-id"]
        start_time = datetime.now()

        printer.print("\n### SYSTEM: making decision based on tool calls", arg_spec)
        printer.print(tool_calls)

        messages = make_decision_messages(tool_calls)
        messages = decision(completionfn, messages, printer)

        printer.print(f"\n### SYSTEM: verifying function i{i}q{idx}")
        verification_result = verify(config, postfn, completionfn, messages, printer, attempt_id, partial(format_inputs, arg_spec))

        time_taken = (datetime.now() - start_time).total_seconds()
        q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id, {"i": i, "v": idx})

    return verification_result

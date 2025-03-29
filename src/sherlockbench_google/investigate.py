import sys
from google.genai import types

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

    function_call_result_message = {"function_response":
                                    {"name": fnname,
                                     "response": {"output": json.dumps(fnoutput)}}}

    return function_call_result_message

def investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec):
    msg_limit = config["msg-limit"]

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
    for count in range(0, msg_limit):
        completion = completionfn(contents=messages, tools=tools)

        message = completion.text if not ValueError else ""
        tool_calls = completion.function_calls

        printer.print("\n--- LLM ---")
        printer.indented_print(message)

        if tool_calls:
            #print(tool_calls[0])

            printer.print("\n### SYSTEM: calling tool")

            next_message = []
            for call in tool_calls:
                next_message.append(handle_tool_call(postfn, printer, attempt_id, call))

                tool_call_counter += 1

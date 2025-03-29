import sys
from google.genai import types
from .utility import save_message

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
    if obj_list.candidates is None:
        print("DEBUG")
        print(obj_list.candidates)
    
    result = ""
    for obj in obj_list.candidates[0].content.parts:
        # Use getattr with a default value to avoid AttributeError
        text = getattr(obj, "text", None)
        if text is not None:
            result += text
    return result


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

    # LLM ran out of messages
    raise MsgLimitException("LLM ran out of messages.")

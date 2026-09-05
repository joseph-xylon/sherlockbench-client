from sherlockbench_client import make_schema

def text_blocks(message):
    """The text blocks of an assistant turn.

    We drop the thinking blocks because verification sends a truncated
    history: a lone turn lifted out of its conversation carries thinking
    signatures whose recorded prefix no longer matches.
    """
    return [b for b in message["content"] if getattr(b, "type", None) == "text"]

def verify(config, postfn, completionfn, eventlogger, messages, printer, attempt_id, v_formatter, make_verification_message):
    # for each verification
    while (v_data := postfn("next-verification", {"attempt-id": attempt_id})):
        verification = v_data["next-verification"]
        output_type = v_data["output-type"]

        verification_formatted = v_formatter(verification)

        printer.print("\n### SYSTEM: inputs:")
        printer.indented_print(verification_formatted)

        # Anthropic 'Requests which include `tool_use` or `tool_result` blocks must define tools.'
        vmessages = [{"role": "assistant", "content": text_blocks(messages[-1])},
                     make_verification_message(verification_formatted)]

        completion = completionfn(messages=vmessages,
                                  output_format=make_schema(output_type))

        if completion.stop_reason == "refusal":
            print("The model refused:", completion.stop_details)

            eventlogger("verify-refusal")
            return False

        if completion.stop_reason == "max_tokens":
            print("The response was truncated.")

            eventlogger("verify-lengtherror")
            return False

        prediction = completion.parsed_output

        # nothing we could parse
        if prediction is None:
            print("No parsed output in the response.")

            eventlogger("verify-jsonerror")
            return False

        thoughts, expected_output = prediction.thoughts, prediction.expected_output

        printer.print("\n--- LLM ---")
        printer.indented_print(thoughts, "\n")
        printer.print()
        printer.indented_print("`" + str(expected_output) + "`\n")

        vstatus = postfn("attempt-verification", {"attempt-id": attempt_id,
                                                  "prediction": expected_output})["status"]

        if vstatus in ("wrong"):
            printer.print("\n### SYSTEM: WRONG")
            return False
        else:
            printer.print("\n### SYSTEM: CORRECT")

        if vstatus in ("done"):
            break

    # if we got here all the verifications passed
    return True

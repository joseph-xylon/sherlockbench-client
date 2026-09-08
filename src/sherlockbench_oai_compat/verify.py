from sherlockbench_client import make_schema

def verify(config, postfn, completionfn, eventlogger, messages, printer, attempt_id, v_formatter, make_verification_message):
    # for each verification
    while (v_data := postfn("next-verification", {"attempt-id": attempt_id})):
        verification = v_data["next-verification"]
        output_type = v_data["output-type"]

        verification_formatted = v_formatter(verification)

        printer.print("\n### SYSTEM: inputs:")
        printer.indented_print(verification_formatted)

        vmessages = messages + [make_verification_message(verification_formatted)]

        completion = completionfn(input=vmessages,
                                  text_format=make_schema(output_type))

        # the responses api doesn't raise for this, it comes back incomplete
        if completion.status == "incomplete":
            print("Response was incomplete:", completion.incomplete_details)

            # well it failed so we return False
            eventlogger("verify-lengtherror")
            return False

        prediction = completion.output_parsed

        # a refusal, or nothing we could parse
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

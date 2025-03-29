import json
from openai import LengthFinishReasonError
from pydantic import BaseModel
from .prompts import make_verification_message
from sherlockbench_client import destructure, make_schema, value_list_to_map

def verify(config, postfn, completionfn, messages, printer, attempt_id):
    # for each verification
    while (v_data := postfn("next-verification", {"attempt-id": attempt_id})):
        verification = v_data["next-verification"]
        output_type = v_data["output-type"]

        printer.print("\n### SYSTEM: inputs:")
        printer.indented_print(verification)

        vmessages = messages + [make_verification_message(value_list_to_map(verification))]

        try:
            completion = completionfn(messages=vmessages,
                                      response_format=make_schema(output_type))
        except LengthFinishReasonError as e:
            print("Caught a LengthFinishReasonError!")
            print("Completion:", e.completion)

            # well it failed so we break
            break

        response = completion.choices[0]

        thoughts, expected_output = destructure(json.loads(response.message.content), "thoughts", "expected_output")

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

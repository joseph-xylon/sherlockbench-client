import sys
from sherlockbench_client import destructure, make_schema, value_list_to_map
from .utility import save_message
from .prompts import make_verification_message

def verify(config, postfn, completionfn, messages, printer, attempt_id):
    # for each verification
    while (v_data := postfn("next-verification", {"attempt-id": attempt_id})):
        verification = v_data["next-verification"]
        output_type = v_data["output-type"]

        printer.print("\n### SYSTEM: inputs:")
        printer.indented_print(verification)

        vmessages = messages + [save_message("user", make_verification_message(value_list_to_map(verification)))]

        completion = completionfn(contents=vmessages,
                                  schema=make_schema(output_type))

        print(completion.text)

        sys.exit(0)

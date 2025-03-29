from google import genai
from google.genai import types
from pprint import pprint

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, start_run, complete_run

#from .investigate import investigate
#from .verify import verify

from datetime import datetime

def save_message(role, text):
    return types.Content(
        role=role,
        parts=[types.Part.from_text(text=text)]
    )

def main():
    config, db_conn, cursor, run_id, attempts, start_time = start_run("google")

    client = genai.Client(api_key=config['api-keys']['google'])

    postfn = lambda *args: post(config["base-url"], run_id, *args)

    contents = []
    while True:
        user_input = input("Enter a string: ")
        
        contents.append(
            save_message("user", user_input)
        )

        response = client.models.generate_content(
            model=config['model'],
            contents=contents
        )
        
        print(response.text)

        contents.append(
            save_message("assistant", response.text)
        )
        
        #print(response.function_calls)
        #print(response.text)

    

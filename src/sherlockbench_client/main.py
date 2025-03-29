import time
import yaml
import copy
import requests
import shutil
import textwrap
from requests import HTTPError
from pydantic import BaseModel
from typing import Callable


def load_config(filepath):
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)

    # Ensure "debug" is always a list
    if "debug" in config:
        if config["debug"] is None:
            config["debug"] = []

    return config

def destructure(dictionary, *keys):
    """it boggles my mind that Python doesn't have destructuring"""
    return (dictionary[key] for key in keys)

def post(base_url, run_id, path, data):
    data["run-id"] = run_id

    try:
        response = requests.post(base_url + path, json=data)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")

        print(response.json().get("error", "no error"))

        if response.status_code == 400 and "error" in response.json():
            return {"output": response.json()["error"]}
        
    return response.json()

def get(base_url, path):
    try:
        response = requests.get(base_url + path)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return {"error": str(http_err)}
        
    return response.json()

class AbortException(Exception):
    """Custom exception for user aborting the operation."""
    pass

def prompt_continue(config, key):
    if key in config["debug"]:
        while True:
            choice = input("Do you want to continue? (y/n): ").strip().lower()
            match choice:
                case 'y' | 'yes':
                    return True
                case 'n' | 'no':
                    raise AbortException("User chose to abort.")
                case _:
                    print("Please enter 'y' for yes or 'n' for no.")

class AccumulatingPrinter:
    def __init__(self):
        # Initialize the internal "megastring"
        self.megastring = ""

    def print(self, *args):
        """
        Prints the concatenated string from the arguments and appends it to the megastring.
        """
        # Concatenate arguments with spaces
        concatenated_string = " ".join(str(arg) for arg in args)
        
        # Print the concatenated string
        print(concatenated_string)
        
        # Append the concatenated string to the megastring
        self.megastring += concatenated_string + "\n"

    def indented_print(self, *args):
        """
        Prints an indented and wrapped version of the concatenated string from the arguments,
        and appends it to the megastring, preserving input newlines.
        """
        # Concatenate arguments with spaces
        concatenated_string = " ".join(str(arg) for arg in args)

        # Get terminal width and calculate wrap width
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        wrap_width = max(terminal_width - 5, 10)  # Ensure wrap width isn't too narrow

        # Split the string into lines to preserve existing newlines
        lines = concatenated_string.splitlines()
        wrapped_lines = []
        for line in lines:
            # Wrap each line individually and prepend indentation
            wrapped_lines.append(textwrap.fill(line, width=wrap_width, subsequent_indent="  ", initial_indent="  "))

        # Combine wrapped lines back into a single string
        indented_string = "\n".join(wrapped_lines)

        # Print the indented string
        print(indented_string)

        # Append the indented string to the megastring
        self.megastring += indented_string + "\n"

    def retrieve(self):
        """
        Returns the accumulated megastring.
        """
        return self.megastring

def make_schema(output_type):
    mapping = {
        "string": str,
        "integer": int,
        "boolean": bool,
        "float": float
    }

    class Prediction(BaseModel):
        """Prediction of the function output."""

        thoughts: str
        expected_output: mapping.get(output_type)

    return Prediction

class LLMRateLimiter:
    def __init__(self, rate_limit_seconds: int, llmfn: Callable, backoff_exceptions: tuple):
        """
        Initialize the RateLimiter.

        :param rate_limit_seconds: The initial number of seconds for the rate limit.
        """
        self.llmfn = llmfn
        self.backoff_exceptions = backoff_exceptions
        self.rate_limit_seconds = rate_limit_seconds
        self.last_call_time = None
        self.total_call_count = 0

    def handle_call(self, llmfn, *args, **kwargs):
        """
        Call the LLM while enforcing the rate limit.
        """

        self.total_call_count += 1

        current_time = time.time()
        if self.last_call_time is not None:
            elapsed_time = current_time - self.last_call_time
            sleep_time = self.rate_limit_seconds - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.last_call_time = time.time()

        try:
            # Call the function
            return llmfn(*args, **kwargs)
        except self.backoff_exceptions as e:
            print()
            print(e)

            self.rate_limit_seconds += 1
            print(f"\n### SYSTEM: backing off for 5 minutes and increasing rate limit to {self.rate_limit_seconds} seconds")
            time.sleep(300)

            self.last_call_time = time.time()
            return llmfn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.handle_call(self.llmfn, *args, **kwargs)

def value_list_to_map(xs):
    """take a vector and return map with alphabetical keys"""
    keys = [chr(97 + i) for i in range(len(xs))]  # Generate keys: 'a', 'b', 'c', etc.
    return dict(zip(keys, xs))

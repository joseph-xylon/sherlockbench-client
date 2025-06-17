import time
import sys
import yaml
import copy
import requests
import shutil
import textwrap
from requests import HTTPError
from pydantic import BaseModel
from typing import Callable
from datetime import datetime
from openai import OpenAI, APITimeoutError, InternalServerError

def load_config(filepath):
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)

    # Ensure "debug" is always a list
    if "debug" in config:
        if config["debug"] is None:
            config["debug"] = []

    return config

def load_provider_config(provider, model_name):
    """Load configuration for the specified provider."""
    config_raw = load_config("resources/config.yaml")

    # only this provider
    config_non_sensitive = {k: v for k, v in config_raw.items() if k != "providers"} | config_raw["providers"][provider][model_name]

    # add credentials
    config = config_non_sensitive | load_config("resources/credentials.yaml")

    return config_non_sensitive, config

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
            if "Invalid exam set:" in response.json()["error"]:
                sys.exit()

            return {"output": response.json()["error"],
                    "error": True}

    # this is how you return a dict with something appended in Python
    return {**response.json(), "error": False}

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
    def __init__(self, rate_limit_seconds: int, llmfn: Callable, backoff_exceptions: list):
        """
        Initialize the RateLimiter.

        :param rate_limit_seconds: The initial number of seconds for the rate limit.
        :param backoff_exceptions: List of tuples, each containing (exception_type, backoff_seconds).
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

        max_retries = 3
        for retry in range(max_retries):
            try:
                # Call the function
                self.last_call_time = time.time()
                return llmfn(*args, **kwargs)

            except Exception as e:
                # Check if this exception matches any of our configured exception-backoff pairs
                backoff_time = None
                for exception_type, backoff_seconds in self.backoff_exceptions:
                    if isinstance(e, exception_type):
                        backoff_time = backoff_seconds
                        break

                # If no matching exception found, re-raise immediately
                if backoff_time is None:
                    raise

                print()
                print(e)

                self.rate_limit_seconds += 1
                print(f"\n### SYSTEM: backing off for {backoff_time} seconds and increasing rate limit to {self.rate_limit_seconds} seconds (retry {retry+1}/{max_retries})")

                # If this was the last retry, re-raise the exception
                if retry == max_retries - 1:
                    raise

                time.sleep(backoff_time)

    def __call__(self, *args, **kwargs):
        return self.handle_call(self.llmfn, *args, **kwargs)

def value_list_to_map(xs):
    """take a vector and return map with alphabetical keys"""
    keys = [chr(97 + i) for i in range(len(xs))]  # Generate keys: 'a', 'b', 'c', etc.
    return dict(zip(keys, xs))

def print_progress_with_estimate(current_index, total_count, start_time):
    """Print progress with estimated time remaining"""
    current_time = datetime.now()
    elapsed = (current_time - start_time).total_seconds()

    if current_index > 1:
        avg_time_per_item = elapsed / (current_index - 1)
        remaining_items = total_count - current_index + 1
        estimated_remaining = avg_time_per_item * remaining_items
        est_hours = int(estimated_remaining // 3600)
        est_minutes = int((estimated_remaining % 3600) // 60)
        est_seconds = int(estimated_remaining % 60)
        time_str = f" (est. {est_hours:02d}:{est_minutes:02d}:{est_seconds:02d} remaining)"
    else:
        time_str = ""

    print(f"\n### SYSTEM: Starting attempt {current_index}/{total_count}{time_str}")

def make_completionfn():
    config_non_sensitive, config = load_provider_config("openai", "o4-mini")

    client = OpenAI(api_key=config['api-keys']['openai'])

    def create_completion(client, **kwargs):
        """closure to pre-load the model"""

        return client.beta.chat.completions.parse(
            **kwargs
        )

    def completionfn(**kwargs):
        if "temperature" in config:
            kwargs["temperature"] = config['temperature']

        if "reasoning_effort" in config:
            kwargs["reasoning_effort"] = config['reasoning_effort']

        return create_completion(client, model=config['model'], **kwargs)

    return LLMRateLimiter(rate_limit_seconds=config['rate-limit'],
                          llmfn=completionfn,
                          backoff_exceptions=[(APITimeoutError, 300),
                                              (InternalServerError, 60)])

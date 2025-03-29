from google import genai
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, start_run, complete_run

from .investigate import investigate
from .verify import verify

from datetime import datetime

def main():
    config, db_conn, cursor, run_id, attempts, start_time = start_run("google")

    client = genai.Client(api_key=config['api-keys']['google'])

    postfn = lambda *args: post(config["base-url"], run_id, *args)

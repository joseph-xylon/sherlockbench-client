import os
from .main import load_config, load_provider_config, destructure, post
from . import queries as q
from datetime import datetime
import argparse
import psycopg2
import re
import traceback
import sys
import uuid
from pprint import pprint
from filelock import FileLock, Timeout
from .run_internal import (
    resume_failed_run,
    start_new_run,
    reset_attempt,
    save_run_failure,
    process_remaining_attempts,
    pick_executor
)

# Global variable to track the current attempt being processed
_current_attempt = None

def set_current_attempt(attempt):
    """Set the current attempt being processed globally"""
    global _current_attempt
    _current_attempt = attempt

def get_current_attempt():
    """Get the current attempt being processed"""
    global _current_attempt
    return _current_attempt

def is_valid_uuid(uuid_string):
    """
    Check if a string is a valid UUID.

    Args:
        uuid_string (str): String to validate as UUID

    Returns:
        bool: True if string is a valid UUID, False otherwise
    """
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
    return bool(uuid_pattern.match(uuid_string))

def start_run(provider):
    """Various things to get the run started:
       - parse the args
       - establish db connection
       - contact the server to start the run
       - add the run info to the db
       - handle resuming from interrupted runs
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SherlockBench with a required argument.")
    parser.add_argument("model_name", help="The name of the model to use for the run")
    parser.add_argument("arg", help="The id of an existing run, or the id of a problem-set.")
    parser.add_argument("--attempts-per-problem", type=int, help="Number of attempts per problem")
    parser.add_argument("--resume", choices=["skip", "retry"], help="How to handle resuming from a failed run: 'skip' the failed attempt, or 'retry' it")
    parser.add_argument("--labels", nargs="+", help="Optional labels for this run (e.g., 'baseline', 'experiment', 'keeper')")

    args = parser.parse_args()

    # Read config
    config_non_sensitive, config = load_provider_config(provider, args.model_name)

    # Connect to postgresql
    db_conn = psycopg2.connect(config["postgres-url"])
    cursor = db_conn.cursor()

    # Check if this is an existing run ID
    is_uuid = is_valid_uuid(args.arg)
    run_id = args.arg if is_uuid else None

    # Handle resuming a failed run or starting a new one
    if is_uuid and args.resume:
        # Resuming a failed run
        run_id, run_type, benchmark_version, attempts = resume_failed_run(config, cursor, run_id, args)
    else:
        # Starting a new run
        run_id, run_type, benchmark_version, attempts = start_new_run(config_non_sensitive, cursor, args, provider, is_uuid, run_id)

    # Update config with important run metadata
    config["run_type"] = run_type
    config["benchmark_version"] = benchmark_version

    # Return unified result regardless of path
    return (config, args.model_name, db_conn, cursor, run_id, attempts, datetime.now())

def complete_run(postfn, db_conn, cursor, run_id, start_time, total_call_count, config):
    run_time, score, percent, problem_names = destructure(postfn("complete-run", {}), "run-time", "score", "percent", "problem-names")

    # we have the problem names now so we can add that into the db
    q.add_problem_names(cursor, problem_names)

    # save the results to the db
    q.save_run_result(cursor, run_id, start_time, score, percent, total_call_count)

    # print the results
    print(f"\n### SYSTEM: run complete for model `{config['model']}`.")
    print(f"\nRun id: {run_id}")
    print(f"\nFinal score: {score['numerator']}/{score['denominator']} ({percent / 100:.0%})")

    # Calculate and display pass@k if we have multiple attempts per problem
    pass_at_k, k, problems_passed, total_problems = q.calculate_pass_at_k(cursor, run_id)

    if k > 1:  # Only display pass@k if we have multiple attempts per problem
        print(f"\nPass@{k} score: {problems_passed}/{total_problems} ({pass_at_k:.0%})")

    # Why do database libraries require so much boilerplate?
    db_conn.commit()
    cursor.close()
    db_conn.close()

def run_with_error_handling(provider, main_function, ex_spec):
    """
    Run a provider's main function with centralized error handling.

    This function handles the entire lifecycle of a benchmark run:
    1. Sets up the run using start_run
    2. Calls the provider's main function
    3. Completes the run when successful
    4. Catches and logs any exceptions to the database

    Args:
        provider: String identifying the provider (e.g., "openai", "anthropic")
        main_function: Function that implements the provider's benchmark logic.
                       It should take (config, db_conn, cursor, run_id, attempts, start_time)
                       and return (postfn, total_call_count, config) for run completion.
    """

    lock_path = f"/tmp/sherlockbench_client_{provider}.lock"
    lock = FileLock(lock_path)

    try:
        lock.acquire(blocking=False)
    except Timeout:
        print("A run for this provider is already in-progress. Awaiting it's completion.")
        print()

    with lock:
        # Start the run
        config, model_name, db_conn, cursor, run_id, attempts, start_time = start_run(provider)

        executor = pick_executor(config, ex_spec)

        try:
            # Call the provider's main function, which should return info needed for completion
            postfn, total_call_count, _ = main_function(executor, config, db_conn, cursor, run_id, attempts, start_time)

            # Complete the run
            complete_run(postfn, db_conn, cursor, run_id, start_time, total_call_count, config)

        except Exception as e:
            # Capture error information
            error_type = type(e).__name__
            error_message = str(e)
            trace_info = traceback.format_exc()

            print(f"\n### SYSTEM ERROR: {error_type}: {error_message}")

            # Save error information to database if we have a connection
            if db_conn and cursor and run_id:
                print("attempts: ", attempts)

                error_info = {
                    "error_type": error_type,
                    "error_message": error_message,
                    "traceback": trace_info
                }

                try:
                    # Get the current attempt from our global tracker
                    current_attempt = get_current_attempt()

                    save_run_failure(cursor, run_id, attempts, current_attempt, error_info)
                    db_conn.commit()

                    # Provide resumption instructions to the user
                    script_name = sys.argv[0].rsplit('/', 1)[-1]
                    print("\n### SYSTEM INFO: Run failed. To resume this run, use one of the following:")
                    print(f"  {script_name} {model_name} {run_id} --resume=skip   # Skip the failed attempt")
                    print(f"  {script_name} {model_name} {run_id} --resume=retry  # Retry the failed attempt")

                except Exception as save_error:
                    print(f"\n### SYSTEM ERROR: Failed to save error information: {save_error}")
                finally:
                    try:
                        cursor.close()
                        db_conn.close()
                    except:
                        pass

            # Re-raise the exception to exit with error
            raise

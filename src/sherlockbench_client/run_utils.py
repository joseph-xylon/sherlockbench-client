from .main import load_config, destructure, post, get
from . import queries as q
import argparse
from datetime import datetime
import psycopg2
import re
import traceback
import sys
import uuid
from pprint import pprint

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

def parse_args():
    """Parse command line arguments for the SherlockBench run."""
    parser = argparse.ArgumentParser(description="Run SherlockBench with a required argument.")
    parser.add_argument("arg", nargs="?", help="The id of an existing run, or the id of a problem-set. Use 'list' to see available problem sets.")
    parser.add_argument("--attempts-per-problem", type=int, help="Number of attempts per problem")
    parser.add_argument("--resume", choices=["skip", "retry"], help="How to handle resuming from a failed run: 'skip' the failed attempt, or 'retry' it")

    args = parser.parse_args()
    
    # Check if arg is missing and print usage
    if args.arg is None:
        parser.print_help()
        print("\nTip: Use 'list' as the argument to see available problem sets.")
        exit(1)
        
    return args

def load_provider_config(provider):
    """Load configuration for the specified provider."""
    config_raw = load_config("resources/config.yaml")
    config_non_sensitive = {k: v for k, v in config_raw.items() if k != "providers"} | config_raw["providers"][provider]
    config = config_non_sensitive | load_config("resources/credentials.yaml")
    return config

def handle_list_command(config):
    """Handle the 'list' command to show available problem sets."""
    response = get(config['base-url'], "problem-sets")
    problem_sets = response.get('problem-sets', {})
    
    print("Available problem sets:")
    print("======================")
    
    for category, problems in problem_sets.items():
        print(f"\n{category}:")
        for problem in problems:
            print(f"  - {problem['name']} :: {problem['id']}")
    
    sys.exit(0)

def resume_failed_run(config, cursor, run_id, args):
    """Resume a previously failed run."""
    # Get the failed run info from the database
    failed_run = q.get_failed_run(cursor, run_id)
    assert failed_run

    print(f"\n### SYSTEM: Found interrupted run with id: {run_id}")

    # Extract key values using destructure
    failure_info, benchmark_version, run_config = destructure(
        failed_run, "failure_info", "benchmark_version", "config"
    )
    
    # Get the failed attempt from the failure info
    failed_attempt = failure_info["current_attempt"]
    assert failed_attempt

    attempt_id = failed_attempt["attempt-id"]

    # Handle resume options
    if args.resume == "retry" and attempt_id:
        if not handle_retry_attempt(config, run_id, attempt_id):
            sys.exit(1)
    elif args.resume == "skip":
        print(f"\n### SYSTEM: Will skip failed attempt: {attempt_id}")

    # Update config with info from the failed run
    config.update(run_config)

    # Get information about the run
    run_type = run_config.get("run_type", "unknown")

    # Get and process remaining attempts
    attempts = process_remaining_attempts(cursor, run_id, failure_info, failed_attempt, args.resume)
    
    print(f"Resuming {run_type} benchmark with run-id: {run_id}")
    
    return run_id, run_type, benchmark_version, attempts

def handle_retry_attempt(config, run_id, attempt_id):
    """Reset a failed attempt on the server for retry."""
    print(f"\n### SYSTEM: Attempting to reset failed attempt: {attempt_id}")
    reset_success = reset_attempt(config, run_id, attempt_id)

    if not reset_success:
        print("\n### SYSTEM ERROR: Failed to reset attempt, exiting.")
        return False
    return True

def process_remaining_attempts(cursor, run_id, failure_info, failed_attempt, resume_mode):
    """Process the list of attempts and filter out completed or skipped ones."""
    # Get a list of already completed attempts
    completed_attempts = q.get_completed_attempts(cursor, run_id)

    all_attempts = failure_info["all_attempts"]

    if not all_attempts:
        print("\n### SYSTEM ERROR: Could not find the list of attempts in failure info")
        print("This may be because the run was started before the update to save all attempts")
        print("Please use the --retry-specific-attempt=<attempt-id> option instead")
        sys.exit(1)

    # Filter out attempts that have already been completed
    attempts = []
    for attempt in all_attempts:
        if attempt["attempt-id"] not in completed_attempts:
            attempts.append(attempt)

    # If we're skipping the failed attempt, remove it from the attempts list
    if resume_mode == "skip":
        attempts = [a for a in attempts if a["attempt-id"] != failed_attempt["attempt-id"]]

    print(f"Found {len(completed_attempts)} completed attempts")
    print(f"Remaining attempts to process: {len(attempts)}")
    
    return attempts

def start_new_run(config, cursor, args, provider, is_uuid, run_id):
    """Start a new benchmark run."""
    subset = config.get("subset")  # none if key is missing
    post_data = {"client-id": f"{provider}/{config['model']}"}

    if subset:
        post_data["subset"] = subset

    if args.attempts_per_problem:
        post_data["attempts-per-problem"] = args.attempts_per_problem

    if is_uuid:
        post_data["existing-run-id"] = run_id
    else:
        post_data["problem-set"] = args.arg
    
    run_id, run_type, benchmark_version, attempts = destructure(
        post(config['base-url'], None, "start-run", post_data),
        "run-id", "run-type", "benchmark-version", "attempts"
    )

    print(f"Starting {run_type} benchmark with run-id: {run_id}")

    # Create the run table entry (only for new runs, not resuming)
    q.create_run(cursor, config, run_id, benchmark_version)
    
    return run_id, run_type, benchmark_version, attempts

def start_run(provider):
    """Various things to get the run started:
       - parse the args
       - load the config
       - establish db connection
       - contact the server to start the run
       - add the run info to the db
       - handle resuming from interrupted runs
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_provider_config(provider)

    # Handle the "list" argument
    if args.arg == "list":
        handle_list_command(config)

    # Connect to postgresql
    db_conn = psycopg2.connect(config["postgres-url"])
    cursor = db_conn.cursor()

    # Check if this is an existing run ID
    is_uuid = is_valid_uuid(args.arg)
    run_id = args.arg if is_uuid else None
    
    # Variables used across different execution paths
    attempts = None
    benchmark_version = None
    run_type = None
    
    # Handle resuming a failed run or starting a new one
    if is_uuid and args.resume:
        # Resuming a failed run
        run_id, run_type, benchmark_version, attempts = resume_failed_run(config, cursor, run_id, args)
    else:
        # Starting a new run
        run_id, run_type, benchmark_version, attempts = start_new_run(config, cursor, args, provider, is_uuid, run_id)
    
    # Update config with important run metadata
    config["run_type"] = run_type
    config["benchmark_version"] = benchmark_version
    
    # Return unified result regardless of path
    return (config, db_conn, cursor, run_id, attempts, datetime.now())

def complete_run(postfn, db_conn, cursor, run_id, start_time, total_call_count, config):
    run_time, score, percent, problem_names = destructure(postfn("complete-run", {}), "run-time", "score", "percent", "problem-names")

    # we have the problem names now so we can add that into the db
    q.add_problem_names(cursor, problem_names)

    # save the results to the db
    q.save_run_result(cursor, run_id, start_time, score, percent, total_call_count)

    # print the results
    print("\n### SYSTEM: run complete for model `" + config["model"] + "`.")
    print("Final score:", score["numerator"], "/", score["denominator"])
    print("Percent:", percent)
    
    # Why do database libraries require so much boilerplate?
    db_conn.commit()
    cursor.close()
    db_conn.close()

def save_run_failure(cursor, run_id, all_attempts, current_attempt, error_info):
    """
    Save information about a run failure to the database.
    
    Args:
        cursor: Database cursor
        run_id: The ID of the run that failed
        current_attempt: Information about the attempt that was in progress during failure,
                        or None if no attempt was in progress
        error_info: Dictionary containing error details (type, message, traceback, etc.)
    """
    # Create a failure info object containing error information and the current attempt
    failure_info = {
        "error_type": error_info.get("error_type", "Unknown"),
        "error_message": error_info.get("error_message", "No message"),
        "traceback": error_info.get("traceback", "No traceback"),
        "current_attempt": current_attempt,
        "all_attempts": all_attempts
    }
    
    # Save the failure info to the database
    q.save_run_failure(cursor, run_id, failure_info)
    
    # Print error information
    print("\n### SYSTEM ERROR: An uncaught exception occurred")
    print(f"Error type: {failure_info['error_type']}")
    print(f"Error message: {failure_info['error_message']}")
    print("The error has been recorded in the database.")
    
def reset_attempt(config, run_id, attempt_id):
    """
    Reset a failed attempt using the API so it can be retried.
    
    Args:
        config: Configuration dictionary containing base URL
        run_id: UUID of the run
        attempt_id: UUID of the attempt to reset
        
    Returns:
        bool: True if the reset was successful, False otherwise
    """
    try:
        # Call the reset-attempt API endpoint
        print(f"\n### DEBUG: Calling reset-attempt API for run_id={run_id}, attempt_id={attempt_id}")
        
        response = post(config["base-url"],
                        str(run_id),
                        "developer/reset-attempt",
                        {"attempt-id": str(attempt_id)})
        
        # Debug information about the response
        print(f"\n### DEBUG: API response type: {type(response)}")
        print(f"\n### DEBUG: API response content: {response}")
        
        # Simple string check for success since we don't know the exact structure
        if "success" in str(response).lower():
            print(f"\n### SYSTEM: Successfully reset attempt {attempt_id}")
            return True
        else:
            print(f"\n### SYSTEM ERROR: Failed to reset attempt. Response: {response}")
            return False
            
    except Exception as e:
        print(f"\n### SYSTEM ERROR: Failed to reset attempt: {str(e)}")
        print(f"\n### DEBUG: Exception traceback: {traceback.format_exc()}")
        return False

def run_with_error_handling(provider, main_function):
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
    config = None
    db_conn = None
    cursor = None
    run_id = None
    attempts = None
    start_time = None
    
    try:
        # Start the run
        config, db_conn, cursor, run_id, attempts, start_time = start_run(provider)
        
        # Call the provider's main function, which should return info needed for completion
        postfn, total_call_count, _ = main_function(config, db_conn, cursor, run_id, attempts, start_time)
        
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
                print("\n### SYSTEM INFO: Run failed. To resume this run, use one of the following:")
                print(f"  sherlockbench_{provider} {run_id} --resume=skip   # Skip the failed attempt")
                print(f"  sherlockbench_{provider} {run_id} --resume=retry  # Retry the failed attempt")
                
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

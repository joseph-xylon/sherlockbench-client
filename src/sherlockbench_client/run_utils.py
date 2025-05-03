from .main import load_config, destructure, post, get
from . import queries as q
import argparse
from datetime import datetime
import psycopg2
import re
import traceback
import sys

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
       - load the config
       - establish db connection
       - contact the server to start the run
       - add the run info to the db
    """

    parser = argparse.ArgumentParser(description="Run SherlockBench with a required argument.")
    parser.add_argument("arg", nargs="?", help="The id of an existing run, or the id of a problem-set. Use 'list' to see available problem sets.")
    parser.add_argument("--attempts-per-problem", type=int, help="Number of attempts per problem")

    args = parser.parse_args()
    
    # Check if arg is missing and print usage
    if args.arg is None:
        parser.print_help()
        print("\nTip: Use 'list' as the argument to see available problem sets.")
        exit(1)

    config_raw = load_config("resources/config.yaml")
    config_non_sensitive = {k: v for k, v in config_raw.items() if k != "providers"} | config_raw["providers"][provider]
    config = config_non_sensitive | load_config("resources/credentials.yaml")

    # Handle the "list" argument - get and print available problem sets
    if args.arg == "list":
        response = get(config['base-url'], "problem-sets")
        problem_sets = response.get('problem-sets', {})
        
        print("Available problem sets:")
        print("======================")
        
        for category, problems in problem_sets.items():
            print(f"\n{category}:")
            for problem in problems:
                print(f"  - {problem['name']} :: {problem['id']}")
        
        exit(0)

    # connect to postgresql
    db_conn = psycopg2.connect(config["postgres-url"])
    cursor = db_conn.cursor()

    subset = config.get("subset")  # none if key is missing
    post_data = {"client-id": f"{provider}/{config['model']}"}

    if subset:
        post_data["subset"] = subset

    if args.attempts_per_problem:
        post_data["attempts-per-problem"] = args.attempts_per_problem

    if is_valid_uuid(args.arg):
        post_data["existing-run-id"] = args.arg

    else:
        post_data["problem-set"] = args.arg
    
    run_id, run_type, benchmark_version, attempts = destructure(post(config['base-url'],
                                                                     None,
                                                                     "start-run",
                                                                     post_data),
                                                                "run-id", "run-type", "benchmark-version", "attempts")

    print(f"Starting {run_type} benchmark with run-id: {run_id}")

    config_non_sensitive["run_type"] = run_type

    # we create the run table now even though we don't have all the data we need yet
    q.create_run(cursor, config_non_sensitive, run_id, benchmark_version)
    
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

def save_run_failure(cursor, run_id, current_attempt, error_info):
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
        "current_attempt": current_attempt
    }
    
    # Save the failure info to the database
    q.save_run_failure(cursor, run_id, failure_info)
    
    # Print error information
    print("\n### SYSTEM ERROR: An uncaught exception occurred")
    print(f"Error type: {failure_info['error_type']}")
    print(f"Error message: {failure_info['error_message']}")
    print("The error has been recorded in the database.")

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
            error_info = {
                "error_type": error_type,
                "error_message": error_message,
                "traceback": trace_info
            }
            
            try:
                # Get the current attempt from our global tracker
                current_attempt = get_current_attempt()
                save_run_failure(cursor, run_id, current_attempt, error_info)
                db_conn.commit()
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

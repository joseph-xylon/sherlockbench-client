import sys
import types
from datetime import datetime
from pprint import pprint

from .main import destructure, post
from . import queries as q

def resume_failed_run(config, cursor, run_id, args):
    """Resume a previously failed run."""
    # Get the failed run info from the database
    failed_run = q.get_failed_run(cursor, run_id)
    assert failed_run

    failure_info, benchmark_version, run_config = destructure(
        failed_run, "failure_info", "benchmark_version", "config"
    )

    failed_attempt = failure_info["current_attempt"]
    attempt_id = failed_attempt["attempt-id"]
    run_type = run_config["run_type"]

    print(f"\n### SYSTEM: Found interrupted run with id: {run_id}")

    # Update config with info from the failed run
    config.update(run_config)

    print("\n### SYSTEM: Using the following config:")
    pprint(config)

    # Handle resume options
    if args.resume == "retry":
        print(f"\n### SYSTEM: Attempting to reset failed attempt: {attempt_id}")
        reset_success = reset_attempt(config, run_id, attempt_id)

        if reset_success:
            print(f"\n### SYSTEM: Successfully reset attempt {attempt_id}")
        else:
            print("\n### SYSTEM ERROR: Failed to reset attempt, exiting.")
            sys.exit(1)

    elif args.resume == "skip":
        print(f"\n### SYSTEM: Will skip failed attempt: {attempt_id}")

        q.fail_attempt(cursor, run_id, attempt_id)

    # Get and process remaining attempts
    attempts = process_remaining_attempts(cursor, run_id, failure_info, failed_attempt, args.resume)

    print(f"Resuming {run_type} benchmark with run-id: {run_id}")

    return run_id, run_type, benchmark_version, attempts

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

def start_new_run(config_non_sensitive, cursor, args, provider, is_uuid, run_id):
    """Start a new benchmark run."""
    subset = config_non_sensitive.get("subset")  # none if key is missing
    model = config_non_sensitive['model']
    post_data = {"client-id": f"{provider}/{model}"}

    if subset:
        post_data["subset"] = subset

    if args.attempts_per_problem:
        post_data["attempts-per-problem"] = args.attempts_per_problem

    if is_uuid:
        post_data["existing-run-id"] = run_id
    else:
        post_data["problem-set"] = args.arg

    run_id, run_type, benchmark_version, attempts = destructure(
        post(config_non_sensitive['base-url'], None, "start-run", post_data),
        "run-id", "run-type", "benchmark-version", "attempts"
    )

    print(f"Starting {run_type} benchmark with model {model}")
    print(f"Run id: {run_id}")

    config_non_sensitive["run_type"] = run_type

    # Add labels to config if provided
    labels = args.labels
    if labels:
        print(f"This run will have labels: {', '.join(labels)}")

    # Create the run table entry (only for new runs, not resuming)
    q.create_run(cursor, config_non_sensitive, run_id, benchmark_version, labels)

    return run_id, run_type, benchmark_version, attempts

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
        "all_attempts": all_attempts,
        "failure_datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        response = post(config["base-url"],
                        str(run_id),
                        "developer/reset-attempt",
                        {"attempt-id": str(attempt_id)})

        # Check for success in status key
        return response.get("status") == "success"

    except Exception as e:
        print(f"\n### SYSTEM ERROR: Failed to reset attempt: {str(e)}")
        return False

def get_or_error(d, key, error_msg):
    if key in d:
        return d[key]
    else:
        sys.exit(error_msg)

def pick_executor(config, ex_spec):
    "Either we have been provided with a function, or a spec to match against the config."
    if isinstance(ex_spec, types.FunctionType):
        return ex_spec

    elif isinstance(ex_spec, dict):
        run_mode = get_or_error(config, "default-run-mode",
                                "ERROR: can't find default-run-mode in config")

        executor = get_or_error(ex_spec, run_mode,
                                "ERROR: specified run-mode not recognized")

        print(f"Using {run_mode} mode")

        return executor

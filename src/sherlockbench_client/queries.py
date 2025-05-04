from pypika import Query, Table, Field
from datetime import datetime
import json
import uuid
from pprint import pprint


def create_run(cursor, config_non_sensitive, run_id, benchmark_version):
    start_time = datetime.now()
    run_data = {"id": run_id,
                "model_identifier": config_non_sensitive["model"],
                "benchmark_version": benchmark_version.split('.', 1)[0],
                "config": json.dumps(config_non_sensitive),
                "datetime_start": start_time.strftime('%Y-%m-%d %H:%M:%S')
                }

    runs = Table("runs")
    insert_query = Query.into(runs).columns(*run_data.keys()).insert(*run_data.values())
    cursor.execute(str(insert_query))
    cursor.connection.commit()

def get_failed_run(cursor, run_id):
    """
    Retrieve information about a failed run from the database.

    Args:
        cursor: Database cursor
        run_id: The UUID of the run to check

    Returns:
        dict: A dictionary containing run information if it exists and has failure_info,
              or None if the run doesn't exist or wasn't interrupted
    """
    runs = Table("runs")

    # Check if the run exists and has failure_info
    query = (
        Query.from_(runs)
        .select(
            runs.id,
            runs.model_identifier,
            runs.benchmark_version,
            runs.config,
            runs.failure_info
        )
        .where(
            (runs.id == str(run_id)) &
            (runs.failure_info.notnull())
        )
    )

    cursor.execute(str(query))
    result = cursor.fetchone()

    if not result:
        return None

    run_id, model_identifier, benchmark_version, config_json, failure_info_json = result

    # The psycopg2 driver already converts JSONB columns to Python dictionaries,
    # so no need to call json.loads()

    return {
        "id": run_id,
        "model_identifier": model_identifier,
        "benchmark_version": benchmark_version,
        "config": config_json,
        "failure_info": failure_info_json
    }

def get_completed_attempts(cursor, run_id):
    """
    Retrieve a list of completed attempts for a run.

    The attempts are only added to the db once completed so this is all of them.
    """
    attempts = Table("attempts")

    query = (
        Query.from_(attempts)
        .select(attempts.id)
        .where(attempts.run_id == str(run_id))
    )

    cursor.execute(str(query))
    results = cursor.fetchall()

    # Extract the attempt IDs
    return [str(result[0]) for result in results]

def add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id):
    attempt_data = {"id": attempt_id,
                    "run_id": run_id,
                    "result": verification_result,
                    "time_taken": time_taken,
                    "tool_calls": tool_call_count,
                    "complete_log": printer.retrieve(),
                    "api_calls": completionfn.total_call_count - start_api_calls}

    insert_query = Query.into(Table("attempts")).columns(*attempt_data.keys()).insert(*attempt_data.values())
    cursor.execute(str(insert_query))
    cursor.connection.commit()

def add_problem_names(cursor, problem_names):
    """problem_names is a list of dicts, each containing 'id' and 'function_name'"""
    attempts = Table("attempts")

    for problem in problem_names:
        update_query = (
            Query.update(attempts)
            .set(attempts.function_name, problem['function_name'])
            .where(attempts.id == problem['id'])
        )
        cursor.execute(str(update_query))

    cursor.connection.commit()

def save_run_result(cursor, run_id, start_time, score, percent, total_call_count):
    runs = Table("runs")

    update_query = (
    Query.update(runs)
         .set(runs.total_run_time, (datetime.now() - start_time).total_seconds())
         .set(runs.final_score, json.dumps({"numerator": score["numerator"], "denominator": score["denominator"]}))
         .set(runs.score_percent, percent)
         .set(runs.total_api_calls, total_call_count)
         .where(runs.id == run_id)
)
    cursor.execute(str(update_query))
    cursor.connection.commit()

def save_run_failure(cursor, run_id, failure_info):
    """
    Save information about a run failure to the database.

    Args:
        cursor: Database cursor
        run_id: The ID of the run that failed
        failure_info: Dictionary containing information about the failure
                     (will be stored as JSON in the database)
    """
    runs = Table("runs")

    update_query = (
        Query.update(runs)
        .set(runs.failure_info, json.dumps(failure_info))
        .where(runs.id == run_id)
    )

    cursor.execute(str(update_query))
    cursor.connection.commit()

def get_attempts_by_function(cursor, run_id):
    """
    Get all attempts for a run, grouped by function_name.

    Args:
        cursor: Database cursor
        run_id: The UUID of the run

    Returns:
        dict: A dictionary with function_name as keys and lists of attempts as values
    """
    attempts = Table("attempts")

    query = (
        Query.from_(attempts)
        .select(
            attempts.function_name,
            attempts.result
        )
        .where(attempts.run_id == str(run_id))
        .orderby(attempts.function_name)
    )

    cursor.execute(str(query))
    results = cursor.fetchall()

    # Group attempts by function_name
    attempts_by_function = {}
    for function_name, result in results:
        if function_name not in attempts_by_function:
            attempts_by_function[function_name] = []
        attempts_by_function[function_name].append(result)

    return attempts_by_function

def calculate_pass_at_k(cursor, run_id):
    """
    Calculate the pass@k metric for a run with multiple attempts per problem.

    Args:
        cursor: Database cursor
        run_id: The UUID of the run

    Returns:
        tuple: (pass@k score, k value, problems_passed, total_problems)
    """
    # Get attempts grouped by function
    attempts_by_function = get_attempts_by_function(cursor, run_id)

    if not attempts_by_function:
        return 0, 0, 0, 0

    # Count problems that have at least one successful attempt
    problems_passed = 0
    total_problems = len(attempts_by_function)

    for attempts in attempts_by_function.values():
        # Check if any attempt for this problem succeeded
        if any(result == "true" for result in attempts):
            problems_passed += 1

    # k is the number of attempts per problem (assuming all problems have the same number)
    k = len(next(iter(attempts_by_function.values()))) if attempts_by_function else 0

    # Calculate pass@k score
    pass_at_k = problems_passed / total_problems if total_problems > 0 else 0

    return pass_at_k, k, problems_passed, total_problems
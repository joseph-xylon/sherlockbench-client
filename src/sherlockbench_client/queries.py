from pypika import Query, Table
from datetime import datetime
import json

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
    cursor.connection.commit() # Commit after creating run

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
    cursor.connection.commit() # Commit after each attempt to ensure data is saved

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
    
    cursor.connection.commit() # Commit after updating problem names

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
    cursor.connection.commit() # Commit after saving run result

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
    
    # Calculate total run time up to the failure
    update_query = (
        Query.update(runs)
        .set(runs.failure_info, json.dumps(failure_info))
        .where(runs.id == run_id)
    )
    
    cursor.execute(str(update_query))
    cursor.connection.commit() # Commit after saving failure info

import argparse
import csv
import io
import sys
from collections import defaultdict

import psycopg2
from pypika import Query, Table, functions

from sherlockbench_client.main import load_config
from sherlockbench_client.run_api import is_valid_uuid

def are_totals_equal(data):
    """
    Check if the 'total' value is the same for all dictionaries in a list.

    Args:
        data (list): A list of dictionaries, each containing a 'total' key.

    Returns:
        $total if all values are equal, False otherwise.
    """
    assert data

    first_total = data[0].get('total')  # Get the 'total' value from the first dictionary
    if all(d.get('total') == first_total for d in data):
        return first_total
    else:
        return False

def get_attempt_summary(cursor, run_ids):
    """
    Get a summary of attempts grouped by function name for specified runs.

    Args:
        cursor: Database cursor
        run_ids: List of UUIDs of the runs to summarize

    Returns:
        dict: A dictionary where keys are function names and values are dicts with 'success' and 'failure' counts
    """
    attempts = Table("attempts")

    # Convert list of UUIDs to strings
    run_id_strings = [str(run_id) for run_id in run_ids]

    # Query the database
    query = (
        Query.from_(attempts)
        .select(
            attempts.function_name,
            attempts.result,
            functions.Count(attempts.id).as_("count")
        )
        .where(attempts.run_id.isin(run_id_strings))
        .groupby(
            attempts.function_name,
            attempts.result
        )
        .orderby(attempts.function_name)
    )

    cursor.execute(str(query))
    results = cursor.fetchall()

    # Process the results
    summary = defaultdict(lambda: {"success": 0, "failure": 0})
    for row in results:
        function_name, result, count = row
        if function_name is None:
            function_name = "Unknown"

        if result.lower() == "true":
            summary[function_name]["success"] += count
        else:
            summary[function_name]["failure"] += count

    return summary


def get_run_ids_by_label(cursor, labels):
    """
    Get run IDs that have any of the specified labels.

    Args:
        cursor: Database cursor
        labels: List of labels to filter by

    Returns:
        list: List of run IDs that match the labels
    """
    run_ids = []

    # Use native SQL query with array overlap operator for better compatibility
    # The && operator checks if there's any overlap between the labels arrays
    query_str = """
    SELECT id FROM runs
    WHERE labels && ARRAY[{}]
    """.format(', '.join([f"'{label}'" for label in labels]))

    cursor.execute(query_str)
    results = cursor.fetchall()

    for result in results:
        run_ids.append(result[0])

    return run_ids


def check_runs_exist(cursor, run_ids):
    """
    Check which runs exist in the database.

    Args:
        cursor: Database cursor
        run_ids: List of UUIDs of the runs to check

    Returns:
        tuple: (existing_ids, missing_ids)
    """
    runs = Table("runs")
    existing_ids = []
    missing_ids = []

    for run_id in run_ids:
        query = (
            Query.from_(runs)
            .select(runs.id)
            .where(runs.id == str(run_id))
        )

        cursor.execute(str(query))
        result = cursor.fetchone()

        if result:
            existing_ids.append(run_id)
        else:
            missing_ids.append(run_id)

    return existing_ids, missing_ids


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Summarize attempt results by function name.")

    # Create a group for mutually exclusive run selection options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-ids", nargs="+", dest="run_ids",
                       help="UUID(s) of the run(s) to summarize")
    group.add_argument("--labels", nargs="+", dest="labels",
                       help="Label(s) to filter runs for summarization (e.g., 'keeper', 'baseline', 'experiment')")

    # Other options
    parser.add_argument("--csv", action="store_true", help="Output in CSV format for spreadsheets")
    parser.add_argument("--sort", action="store_true", help="Sort results by success rate (descending)")

    args = parser.parse_args()
    run_ids = args.run_ids if args.run_ids else None
    labels = args.labels if args.labels else None
    output_csv = args.csv
    sort_results = args.sort

    # If run_ids are provided, validate all of them are UUIDs
    invalid_uuids = []
    if run_ids:
        for run_id in run_ids:
            if not is_valid_uuid(run_id):
                invalid_uuids.append(run_id)

        if invalid_uuids:
            print(f"Error: The following IDs are not valid UUIDs: {', '.join(invalid_uuids)}")
            sys.exit(1)

    try:
        # Load configuration to get database connection info
        config = load_config("resources/credentials.yaml")

        # Connect to postgresql
        db_conn = psycopg2.connect(config["postgres-url"])
        cursor = db_conn.cursor()

        # Determine run IDs based on input
        if labels:
            # Get run IDs that match the specified labels
            run_ids = get_run_ids_by_label(cursor, labels)
            if not run_ids:
                label_list = ', '.join(f"'{label}'" for label in labels)
                print(f"Error: No runs found with label(s) {label_list}")
                sys.exit(1)
            print(f"Found {len(run_ids)} runs with the specified label(s)")

        # Check which runs exist in the database
        existing_ids, missing_ids = check_runs_exist(cursor, run_ids)

        # Exit if any specified run IDs are missing
        if missing_ids:
            print(f"Error: The following run IDs were not found in the database: {', '.join(missing_ids)}")
            sys.exit(1)

        if not existing_ids:
            print("Error: No run IDs were found in the database.")
            sys.exit(1)

        # Get attempt summary for existing runs
        summary = get_attempt_summary(cursor, existing_ids)

        # Output header
        if labels:
            label_list = ', '.join(f"'{label}'" for label in labels)
            print(f"Attempt summary for runs with label(s) {label_list}:")
        elif len(existing_ids) == 1:
            print(f"Attempt summary for run: {existing_ids[0]}")
        else:
            print(f"Attempt summary for {len(existing_ids)} runs:")

            for run_id in existing_ids:
                print(f"  - {run_id}")

        print()

        # Calculate totals and prepare sorted data
        if summary:
            total_success = 0
            total_failure = 0

            # Process the data and sort by success rate
            sorted_data = []
            for function_name, counts in summary.items():
                success = counts["success"]
                failure = counts["failure"]
                total = success + failure
                success_rate = success / total * 100 if total > 0 else 0

                total_success += success
                total_failure += failure

                sorted_data.append({
                    "function_name": function_name,
                    "success": success,
                    "failure": failure,
                    "total": total,
                    "success_rate": success_rate
                })

            # Sort by success rate if requested, otherwise sort by function_name
            if sort_results:
                sorted_data = sorted(sorted_data, key=lambda x: x["success_rate"], reverse=True)
            else:
                sorted_data = sorted(sorted_data, key=lambda x: x["function_name"])

            grand_total = total_success + total_failure

            # Output as CSV if requested
            if output_csv:
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["Function Name", "Success", "Failure", "Total", "Success Rate"])

                for row in sorted_data:
                    writer.writerow([
                        row["function_name"],
                        row["success"],
                        row["failure"],
                        row["total"],
                        f"{row['success_rate']:.0f}"
                    ])

                writer.writerow(["TOTAL", total_success, total_failure, grand_total, ""])

                print(output.getvalue())
            else:
                # Output in table format
                print("{:<40} {:>10} {:>10} {:>10} {:>10}".format(
                    "Function Name", "Success", "Failure", "Total", "Success Rate"
                ))
                print("-" * 85)

                for row in sorted_data:
                    print("{:<40} {:>10} {:>10} {:>10} {:>9.0f}%".format(
                        row["function_name"],
                        row["success"],
                        row["failure"],
                        row["total"],
                        row["success_rate"]
                    ))

                print("-" * 85)
                print("{:<40} {:>10} {:>10} {:>10} {:>10}".format(
                    "TOTAL", total_success, total_failure, grand_total, ""
                ))

                print()
                print(f"Over-all score: {100 / grand_total * total_success:.0f}%")
                equal_per_problem = are_totals_equal(sorted_data)
                if equal_per_problem:
                    passed_once = sum(1 for d in sorted_data if d.get('success', 0) > 0)
                    print(f"pass@{equal_per_problem}: {100 / len(sorted_data) * passed_once:.0f}%")

        else:
            print("\nNo attempts found for the specified run(s).")

        # Show missing runs if any
        if missing_ids:
            print("\nThe following run IDs were not found in the database:")
            for run_id in missing_ids:
                print(f"  - {run_id}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'db_conn' in locals() and db_conn:
            db_conn.close()


if __name__ == "__main__":
    main()

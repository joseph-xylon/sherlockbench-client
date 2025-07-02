The purpose of this branch is to isolate the investigation performance. It runs
the investigation as usual, then always use OpenAI's o4-mini for the decicion
and verification.

o4-mini is hardcoded in `src/sherlockbench_client/main.py`.

There are no special requirements to run this branch, as-long as o4-mini is
configured in your `config.yaml`.

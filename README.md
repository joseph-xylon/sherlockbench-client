> [!IMPORTANT]
> This is a modified version of SherlockBench for doing specific experiments. See the [branch-specific README](README-inv-vs-dv.md).

# SherlockBench Clients

This codebase contains the clients for the SherlockBench AI benchmarking system.

There is a Python package for each LLM provider, sharing as much code as is
reasonable but allowing to accommodate the idiosyncrasies for each provider.

Essentially the clients are sitting in-between two APIs. The LLM provider's API
and the SherlockBench API.

## Main Website
The project homepage: https://sherlockbench.com

## Running
If you want to run this benchmark yourself, you will need:
- An account and API key for whichever LLM provider you want to use
- A computer to install Python and PostgreSQL on. Postgres is how it stores analytics for each run

General instructions follow. Alternatively you may watch this video for Ubuntu instructions: [Installing SherlockBench Client](https://youtu.be/8CdX0VheL5M).

Checkout this code.

Install these:
- PostgreSQL
  - server
  - client
  - libpq-dev
- Python3:
  - runtime
  - pip
  - virtualenv

Create a postgresql database and user.

Create a couple of config files:

A `resources/config.yaml` looks like this:
```
---

# the base URL of the SherlockBench API
#base-url: "http://0.0.0.0:3000/api/"
base-url: "https://api.sherlockbench.com/api/"

providers:
  openai:
    GPT-4o:
      rate-limit: 10
      default-run-mode: "3-phase"

      model: "gpt-4o-2024-08-06"
      #temperature: 0.5

    GPT-4.1:
      rate-limit: 10
      default-run-mode: "3-phase"

      model: "gpt-4.1-2025-04-14"

    o3:
      rate-limit: 10
      default-run-mode: "2-phase"

      model: "o3-2025-04-16"
      reasoning_effort: "medium"
      service_tier: "auto"

    o4-mini:
      rate-limit: 10
      default-run-mode: "2-phase"

      model: "o4-mini-2025-04-16"
      reasoning_effort: "medium"
      service_tier: "auto"

  anthropic:
    Haiku-3.5:
      rate-limit: 20
      default-run-mode: "2-phase"

      model: "claude-3-5-haiku-20241022"
      #temperature: 0.8

    Sonnet-4:
      rate-limit: 120
      default-run-mode: "2-phase"

      model: "claude-sonnet-4-20250514"

    Opus 4:
      rate-limit: 120
      default-run-mode: "2-phase"

      model: "claude-opus-4-20250514"

    Sonnet-4+thinking:
      rate-limit: 120
      default-run-mode: "2-phase"

      model: "claude-sonnet-4-20250514+thinking"

    Opus-4+thinking:
      rate-limit: 120
      default-run-mode: "2-phase"

      model: "claude-opus-4-20250514+thinking"

  google:
    Gemini-2.5-flash:
      rate-limit: 20
      default-run-mode: "3-phase"

      model: "gemini-2.5-flash-preview-05-20"
      #temperature: 0.0

    Gemini-2.5-pro:
      rate-limit: 100
      default-run-mode: "3-phase"

      model: "gemini-2.5-pro-preview-05-06"
      #temperature: 0.0

  xai:
    Grok-3:
      rate-limit: 10
      default-run-mode: "2-phase"

      model: "grok-3"

    Grok-3-mini:
      rate-limit: 10
      default-run-mode: "2-phase"

      model: "grok-3-mini"
      reasoning_effort: "high"

  deepseek:
    v3:
      rate-limit: 30
      default-run-mode: "3-phase"

      model: "deepseek-chat"

    R1:
      rate-limit: 30

      model: "deepseek-reasoner"

```

And a `resources/credentials.yaml` containing your db credentials and API keys:
```
---

postgres-url: "postgresql://user:password@localhost/dbname"
api-keys:
  anthropic: ""
  openai: ""
  google: ""
  fireworks: ""
  xai: ""
  deepseek: ""
  mistral: ""
```

Running it should be essentially:
- make a virtualenv and activate it
- install sherlockbench into your virtualenv with `pip install -e .`
- run `alembic upgrade head` to create the database tables
- type the name of the provider entry-point to run the benchmark (see setup.cfg)

Here is an example of how to run it:
```
# list available problem-sets
sbench_list

# run the benchmark
sbench_anthropic Haiku-3.5 sherlockbench.sample-problems/easy3
```

If you want a breadown per problem, you will want to run multiple attempts per problem:
```
sbench_anthropic Haiku-3.5 sherlockbench.sample-problems/easy3 --attempts-per-problem 10

# summarize the attempts
summarize_attempts --run-ids b92c2ca4-6126-412e-a703-9d3991e99b77
```

## Database Analysis
There are two tables in the database;
- runs stores general information about the test run and it's results
- attempts stores the logs for the individual attempts and some metadata

There are also some views for convenience. These just show the most commonly used columns.
```
select * from runs_view;
select * from attempts_view;
```

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

General instructions follow. Alternatively you may watch this video for Ubuntu instructions: [Installing SherlockBench Client](https://youtu.be/qNIXQTtuFYs).

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

A `resources/config.yaml` should look like this (uncomment sections as appropriate):
```
---

# the base URL of the SherlockBench API
#base-url: "http://0.0.0.0:3000/api/"
base-url: "https://api.sherlockbench.com/api/"

providers:
  openai:
    rate-limit: 10

    model: "gpt-4.1-mini-2025-04-14"
    #model: "o4-mini-2025-04-16"
    #reasoning_effort: "medium"  # low, medium or high
    
    #temperature: 0.5

anthropic:
    rate-limit: 10

    model: "claude-3-5-haiku-20241022"
    # special postfix +thinking enables Anthropic's "extended thinking"
    #model: "claude-sonnet-4-20250514+thinking"
    #model: "claude-opus-4-20250514+thinking"
    
    #temperature: 0.8

  google:
    rate-limit: 10

    #model: "gemini-2.0-flash"
    model: "gemini-2.5-flash-preview-05-20"
    #model: "gemini-2.5-pro-preview-05-06"
    #temperature: 0.0

  fireworks:
    rate-limit: 10

    model: "accounts/fireworks/models/llama-v3p1-405b-instruct"
    #temperature: 0.9
    
    # Qwen recommended settings: https://huggingface.co/Qwen/Qwen3-235B-A22B
    model: "accounts/fireworks/models/qwen3-235b-a22b"
    max_tokens: 32768
    temperature: 0.6
    extra_body:
      top_p: 0.95
      top_k: 20
      min_p: 0

  xai:
    rate-limit: 10

    model: "grok-3-beta"
    #model: "grok-3-mini-beta"
    #reasoning_effort: "high"  # high or low

  deepseek:
    rate-limit: 30

    model: "deepseek-chat"

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

## Database Analysis
There are two tables in the database;
- runs stores general information about the test run and it's results
- attempts stores the logs for the individual attempts and some metadata

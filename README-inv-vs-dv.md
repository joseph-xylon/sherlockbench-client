The purpose of this branch is to help see how dependant the model's performance
on the decision and verification phases is upon the investigation phase.

We run the investigation phase 10 times. Then for each of those, we complete the
test a further 20 times each to get a success percentage for that investigation.

This must be run in a particular way. First you must be running your own
instance of the sherlockbench-api, with a custom problem-set.

This problem-set must contain exactly one problem. Here is an example which can
be added to `config.edn` inside `:custom-problem-sets`:
```
"add-big-small" {:names [("sherlock1" "add biggest and smallest")]}
```

Then you must run sherlockbench-client with 200 attempts per problem. This is
because it will run the investigation 10 times and then complete the attempt 20
times for each so 10*20 = 200.
```
sbench_list
sbench_openai_3p GPT-4.1 custom-addbigsmall --attempts-per-problem 200 --labels 'random'
```

Once the run is complete, this query will help you to extract the results from the database (replace run_id):
```
SELECT
       (meta ->> 'i')::int AS i_value,   -- cast to int so it sorts numerically
       COUNT(*)            AS hits
FROM   attempts
WHERE  result = 'true'
  AND  run_id = 'fde83a5e-2b8b-4482-b49a-31478257173b'
  AND  meta ->> 'i' IN ('0','20','40','60','80','100','120','140','160','180')
GROUP  BY i_value
ORDER  BY i_value;
```

n.b. resuming failed runs is not supported on this branch. If it fails you will
have to restart. n.b. if there are intermittent exceptions you can handle them
with the `backoff_exceptions` setting in each client's main.py.

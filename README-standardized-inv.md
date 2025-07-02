The purpose of this branch is to use a use a pre-recorded set of
investigations. This allows us to measure model's performance on the decision
and verification phases in isolation.

This works by reading in `resources/investigations.yaml` for each problem.

Here is an abridged example of how this looks:
```
problems:
  'add biggest and smallest': |
    (1, 2, 3) → 4
    (5, 10, 4) → 14
  'all odd or even': |
    (1, 2, 3) → False
    (2, 3, 1) → False
  'count consonants': |
    "abc" → 2
    "abcd" → 3
```

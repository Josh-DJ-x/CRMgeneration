# Judge response format

Shared example task: **"Collect 5 coins, then a red key, then reach the goal while avoiding hazards."**

Desired trace: `COIN x 5 -> RED_KEY -> GOAL`.
The generated CRM is shown in [generatorformat.md](generatorformat.md).

In iterative mode, an unsuccessful evaluation is sent to the judge with the CRM,
desired trace, actual agent trace, and raw run information. The judge returns the
three sections below. Copy both the supplied agent trace and run-information JSON
without changing their values; analysis belongs only in `REFLECTION`.

## Example response

This is an **illustrative evaluation, not a recorded experiment**. Assume a grid
with a 32-step shortest valid route. The agent collected five coins and then
reached the 200-step limit without collecting the key or reaching the goal.
The values use the reward table in `generatorformat.md`: five coin rewards and
195 ordinary-step penalties give `5 * 0.1 - 195 * 0.01 = -1.45`.
The reference route gives `5 * 0.1 + 1 + 10 - 25 * 0.01 = 11.25`.

```text
<<<AGENT_TRACE>>>
["COIN", "COIN", "COIN", "COIN", "COIN"]

<<<RUN_INFORMATION>>>
{
  "trace": [
    "COIN",
    "COIN",
    "COIN",
    "COIN",
    "COIN"
  ],
  "episode_reward": -1.45,
  "minimum_reward": -11.99,
  "maximum_reward": 11.25,
  "reward_efficiency": 0.453528,
  "steps": 200,
  "minimum_steps": 32,
  "step_efficiency": 0.16,
  "terminated": false,
  "truncated": true,
  "bell_rings": []
}

<<<REFLECTION>>>
Task: Collect five coins, then a red key, then reach the goal while avoiding hazards.

Trace comparison: The five-coin prefix is correct. RED_KEY and GOAL are missing;
there are no unexpected events in the supplied trace.

Performance: Evaluation ended at the 200-step limit, rather than a terminal state.
Step efficiency is 32 / 200 = 0.16. Reward efficiency is approximately 0.453528.
The exact trace, step-efficiency, and reward-efficiency requirements are not met.

Analysis: The supplied CRM advances to State 1 on the fifth coin and permits
RED_KEY next. The observed failure does not demonstrate a counter or ordering
error. The agent did not learn to reach the key in this evaluation; the trace
alone does not establish whether exploration, layout, or reward scaling caused it.

Suggested revision: Preserve the counters, state transitions, required event
order, and wrong-order penalties. One candidate experiment is to increase the
RED_KEY reward in State 1 from 1.0 to 2.0 while leaving other rewards unchanged.
This may strengthen the signal for reaching the key, but is not guaranteed to
resolve the learning failure. Reevaluate the resulting CRM rather than assuming
that a larger reward fixes it.
```

## How the system uses feedback

- The judge diagnoses the result and suggests changes; the generator writes the
  replacement CRM. The judge does not change the grid or train the agent.
- A mismatch is not the only reason for feedback. A matching trace can still fail
  the configured efficiency requirements. Normally, success requires exact trace
  equality, step efficiency **greater than 0.80**, and reward efficiency **greater
  than 0.99**. These thresholds are configured in `system_loop.py`.
- When the route baseline is unavailable, efficiency fields are `null`, not zero.
  Qualification then depends on trace match alone; do not invent missing metrics
  or require efficiency thresholds for that run.
- If generation validation fails after corrections, judging may occur before any
  agent evaluation. Diagnose the supplied validation error; an empty trace in
  that case is not evidence of an agent learning failure.
- Later judgements should assess the previous suggested change against the new
  result. Preserve correct task logic and distinguish observations from hypotheses.

The reflection is passed to the generator for refinement. The actual agent trace
comes from the evaluation, independently of the judge's copied trace. Keep the
markers and raw JSON sections even though the loop extracts the reflection for
its saved feedback. Both prompt lengths use this response structure.

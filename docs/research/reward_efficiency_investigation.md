# Reward efficiency investigation

Investigated task 37, run `3959d237813743d0b77a5fb381c8b6fb`.

## Calculation

`reward_efficiency()` in `system_functions.py` calculates:

```
(episode_reward - minimum_reward) / (maximum_reward - minimum_reward)
```

If both bounds are equal, it returns 1 when the reward reaches the maximum,
otherwise 0. The score is not clipped.

`calculate_trace_reward_bounds()` finds the shortest physically attainable
ordered route and replays every movement and task event through the CRM to
calculate the reference maximum reward. The minimum is a baseline derived
from transition rewards and the episode step limit, rather than a search for
the worst attainable route.

Evaluation sums the original CRM rewards without discounting. Reward shaping
affects Q-learning updates but is not included in the evaluation score. The
cross-product environment uses CRM rewards rather than the ground environment's
movement rewards.

## Task 37

- Desired trace: 14 bell rings, 4 coins, GOAL.
- Shortest valid route: 44 steps (verified by replaying the route search).
- Agent route: 48 steps.
- Each bell and coin event pays 0.1; GOAL pays 1.0.
- Ordinary movement pays 0.0 in all three CRM states.
- Both routes earn `14 * 0.1 + 4 * 0.1 + 1.0 = 2.8`.
- Minimum reward: 0.0. Maximum reward: 2.8.
- Reward efficiency: `(2.8 - 0) / (2.8 - 0) = 1.0`.

The four extra steps have no reward cost. Thus a reward-optimal route can be
longer than the shortest route. This is consistent with the current formula.

## Options

Use a separate route efficiency metric, `shortest_valid_steps / actual_steps`,
only for runs whose complete trace matches the desired trace. Task 37 would
score `44 / 48 = 0.9167`. This measures physical route length independently of
the generated reward scale and can be used as an additional stopping criterion.

Alternatively, require a negative ordinary-movement reward in every generated
CRM state. For this task, -0.01 on non-event steps would yield a shortest-route
reward of 2.55 and an actual reward of 2.51, so the longer route would no longer
score 1.0. The normalized reward score would still depend on the chosen event
reward scale and minimum bound, so a 0.94 threshold does not guarantee a route
within 6% of the shortest length.

No training or reward metric definitions were changed in this investigation.

# Generator response format

Shared example task: **"Collect 5 coins, then a red key, then reach the goal while avoiding hazards."**

The generator defines the required event order, the physical objects, and the CRM.
Python generates the grid and trains the agent. The response below follows the
markers and grammar used by the current prompts and parser; the Markdown fence
is for documentation only and should not be part of a model response.

## Initial response

```text
<<<DESIRED_TRACE>>>
COIN x 5 -> RED_KEY -> GOAL

<<<ENVIRONMENT>>>
{
  "coins": 5,
  "keys": {
    "red": 1,
    "blue": 0,
    "green": 0,
    "yellow": 0
  },
  "bells": 0,
  "hazards": true
}

<<<CRM>>>
Initial State
u0 = 0

Initial Counters
c0 = (4,)

State Transitions

State 0
    COIN / (NZ) -> 0
    COIN / (Z) -> 1
    RED_KEY or BLUE_KEY or GREEN_KEY or YELLOW_KEY or BELL or GOAL or HAZARD / (-) -> -1
    / (-) -> 0

State 1
    RED_KEY / (-) -> 2
    COIN or BLUE_KEY or GREEN_KEY or YELLOW_KEY or BELL or GOAL or HAZARD / (-) -> -1
    / (-) -> 1

State 2
    GOAL / (-) -> -1
    COIN or RED_KEY or BLUE_KEY or GREEN_KEY or YELLOW_KEY or BELL or HAZARD / (-) -> -1
    / (-) -> 2

Counter Transitions

State 0
    COIN / (NZ) -> (-1,)
    COIN / (Z) -> (0,)
    RED_KEY or BLUE_KEY or GREEN_KEY or YELLOW_KEY or BELL or GOAL or HAZARD / (-) -> (0,)
    / (-) -> (0,)

State 1
    RED_KEY / (-) -> (0,)
    COIN or BLUE_KEY or GREEN_KEY or YELLOW_KEY or BELL or GOAL or HAZARD / (-) -> (0,)
    / (-) -> (0,)

State 2
    GOAL / (-) -> (0,)
    COIN or RED_KEY or BLUE_KEY or GREEN_KEY or YELLOW_KEY or BELL or HAZARD / (-) -> (0,)
    / (-) -> (0,)

Reward Transitions

State 0
    COIN / (NZ) -> 0.1
    COIN / (Z) -> 0.1
    RED_KEY or BLUE_KEY or GREEN_KEY or YELLOW_KEY or BELL or GOAL or HAZARD / (-) -> -10.0
    / (-) -> -0.01

State 1
    RED_KEY / (-) -> 1.0
    COIN or BLUE_KEY or GREEN_KEY or YELLOW_KEY or BELL or GOAL or HAZARD / (-) -> -10.0
    / (-) -> -0.01

State 2
    GOAL / (-) -> 10.0
    COIN or RED_KEY or BLUE_KEY or GREEN_KEY or YELLOW_KEY or BELL or HAZARD / (-) -> -10.0
    / (-) -> -0.01

```

## How the example works

- `DESIRED_TRACE` uses the current compact format. Python expands it to
  `["COIN", "COIN", "COIN", "COIN", "COIN", "RED_KEY", "GOAL"]` for comparison.
  Counts are positive integers; the trace ends with one GOAL and excludes movement.
- `ENVIRONMENT` describes counts, not positions. `hazards` is a Boolean; Python
  chooses hazard positions and their number. `bells` counts physical bells, not
  rings: one bell can support repeated interactions. The goal and agent are implicit.
- The counter starts at `5 - 1 = 4`. Guards test the value **before** the update.
  The first four coins decrement it; the fifth coin advances to State 1.
- State 1 accepts the red key; State 2 accepts the final goal. Wrong-order events
  and hazards terminate with a penalty. A terminal state therefore does not by
  itself establish success: the system also checks the exact event trace.
- Specific transitions precede the fallback `/ (-)`. Counter updates are additive
  integer tuples, such as `(-1,)`; they are not Python expressions. Every state
  uses identical expressions across its state, counter, and reward tables.

Rewards are illustrative choices, not fixed system constants. Ordinary movement
has a negative reward and leaves counters unchanged. No coordinates, walls,
movement actions, or route instructions belong in the generated specification.

## Corrections and refinements

After a valid initial trace and environment are established, they remain fixed,
as does the generated grid. A CRM correction or judge-guided refinement returns
only `<<<CRM>>>` followed by the **complete replacement CRM**, using the same five
headings above. Do not send a patch, repeat the initial sections, or change the task.

If the initial response cannot be parsed or its environment is invalid, the system
requests all three sections again until those initial definitions are established.
Unsupported, contradictory, or unachievable tasks are rejected with only
`<<<INVALID_TASK>>>`; supported short tasks and an omitted explicit goal are valid.

See [judgeformat.md](judgeformat.md) for feedback on this same example.
The live long/short prompts are in `system_specifications.py`; these files document
the response contract and do not configure the prompts.

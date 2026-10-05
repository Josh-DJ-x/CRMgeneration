"""Prompt and format specifications used by the CRM generation loop."""

GENERATOR_CONTEXT = """
You are generating a Counting Reward Machine (CRM) for a fixed 10x10 GridWorld environment.
The agent can move UP, DOWN, LEFT, RIGHT, and INTERACT.
The environment contains collectible RED_KEY, BLUE_KEY, GREEN_KEY, YELLOW_KEY, COIN,
and BELL objects; a GOAL; HAZARD cells; and impassable WALL cells.
The CRM must represent only the desired event trace and its temporal relationships, never
the physical map, object coordinates, paths, or layout. Available event labels are
RED_KEY, BLUE_KEY, GREEN_KEY, YELLOW_KEY, COIN, BELL, GOAL, and HAZARD. Use
counters only where the task requires counting. Do not invent, remove, or reorder trace
events required by the user's task. The agent must always move to the GOAL after completing the task.
"""

CRM_GENERATION_PROMPT = """
## CRM Generation Instructions

You are generating a **Counting Reward Machine (CRM)** for a GridWorld task.
The CRM describes task logic only, never object positions, paths, or movement.

### GridWorld Event Vocabulary
`RED_KEY`, `BLUE_KEY`, `GREEN_KEY`, `YELLOW_KEY`, `COIN`, `BELL`, `GOAL`, and `HAZARD`
are generated automatically by the GridWorld labelling function.

### CRM Components
Every CRM must contain an initial state (`u0`), initial counters (`c0`), state
transition function, counter transition function, and reward transition function.

### Transition Format
Every transition must use `EVENT_EXPRESSION / COUNTER_CONDITION`, for example:
`RED_KEY / (-)`, `GOAL / (-)`, `COIN / (-)`, `BELL  / (NZ)`,
or `GOAL / (Z)`.

Use Boolean `and`, `or`, `not`, and parentheses in event expressions. Counter
conditions are `(-)` (ignore counters), `(Z)` (zero), `(NZ)` (non-zero), or a
comma-separated condition such as `(Z,NZ)` for multiple counters. Counters are used
only when the task requires counting.

In the first generation, output `<<<DESIRED_TRACE>>>`, `<<<ENVIRONMENT>>>`, and
`<<<CRM>>>` sections. DESIRED_TRACE is a compact ordered sequence using `EVENT`
or `EVENT x COUNT` stages joined by `->`, for example:
`COIN x 12 -> BELL x 40 -> GREEN_KEY -> GOAL`.
Counts must be positive integers, omitting a count means one event. End with
exactly one GOAL. Do not include movement, no-op, or other non-event steps in the trace.
In every refinement, output only `<<<CRM>>>`: the system retains the initial desired
trace and environment specification, and they must not be changed.
Do not generate Python code; generate only the CRM specification.
"""

ENVIRONMENT_CONTEXT = """
ENVIRONMENT VARIABLE SPECIFICATION

This applies to the `<<<ENVIRONMENT>>>` section in the first generation only. Specify
the minimum number of objects needed to make the task possible, describing WHAT must
exist rather than WHERE. The Python environment generator decides all positions,
walls, layout, and starting position.

Specify counts for coins, each key colour (red, blue, green, yellow), bells, and
hazards. For counting requirements, provide enough objects to satisfy the requirement.
Do not specify coordinates or wall locations.
"""

ENVIRONMENT_SEMANTICS = """
ENVIRONMENT SEMANTICS

The task takes place in a procedurally generated 10x10 GridWorld. The agent moves one
cell at a time using UP, DOWN, LEFT, RIGHT, and INTERACT.

Grid symbols: A = agent start; . = empty traversable cell; # = impassable wall;
G = goal/exit; H = hazard; C = coin; S = bell; R = red key; B = blue key;
E = green key; Y = yellow key.

Entering a key cell collects the key, entering a coin collects one coin, entering a
hazard produces HAZARD, entering the goal produces GOAL, and using INTERACT while
standing on a bell produces BELL. Walls block movement and produce no events.
The generator decides object placement; the CRM defines temporal task logic only.
"""

REFINEMENT_CONTEXT = """
Use the trace comparison, raw run information, and reflection to improve the CRM. The
desired trace and environment specification are fixed from the first generation and
are not to be repeated or changed. Output only a corrected CRM. If the observed agent
trace differs from the desired trace, update the CRM so the exact event sequence is
produced. Preserve parts of the CRM that already produce the correct ordered trace prefix.
"""

GENERATION_FORMAT = """
IMPORTANT: Respond exactly in this format:

<<<DESIRED_TRACE>>>
COIN x 5 -> RED_KEY -> GOAL

<<<ENVIRONMENT>>>
{
  "coins": 5,
  "keys": {"red": 1, "blue": 0, "green": 0, "yellow": 0},
  "bells": 0,
  "hazards": false
}

<<<CRM>>>
Initial State
u0 = 0

Initial Counters
c0 = (4,)

State Transitions
State 0
    COIN / (-) -> 0
    RED_KEY / (NZ) -> 1
    / (-) -> 0
State 1
    GOAL / (-) -> -1
    / (-) -> 1

Counter Transitions
State 0
    COIN / (-) -> (-1,)
    RED_KEY / (NZ) -> (0,)
    / (-) -> (0,)
State 1
    GOAL / (-) -> (0,)
    / (-) -> (0,)

Reward Transitions
State 0
    COIN / (-) -> 0.1
    RED_KEY / (NZ) -> 1.0
    / (-) -> -0.01
State 1
    GOAL / (-) -> 10.0
    / (-) -> -0.01

DESIRED_TRACE and ENVIRONMENT must appear only in the
first generation. Refinements contain only CRM and must preserve the original trace
and environment. Do not include coordinates, walls, or the agent in ENVIRONMENT. The
Python generator determines layout. If a user specified task is ambiguous, decide on 
one interpretation and make sure the desired trace, environment specification and crm 
all align with the decided interpretation.
"""

REFLECTION_FORMAT = """
You are the judge. You will receive an agent-run event trace. Copy it exactly and give
concise feedback about how it differs from the generator's desired trace. Respond
exactly as:

<<<AGENT_TRACE>>>
["RED_KEY", "BLUE_KEY", "COIN", "COIN", "COIN", "COIN", "COIN", "GOAL"]

<<<RUN_INFORMATION>>>
{...copy the supplied raw run-information JSON exactly...}

<<<REFLECTION>>>
...
"""

FAIL_PARSE = """The CRM failed to parse. Review the correct CRM syntax and fix it."""

CRM_FORMAT = """
A Counting Reward Machine (CRM) has an initial machine state, initial counter
configuration, state transition function, counter transition function, and reward
transition function. Events are produced by the GridWorld labelling function.

Use `EVENT_EXPRESSION / COUNTER_CONDITION`, such as `RED_KEY / (-)`, `COIN / (NZ)`,
`GOAL / (-)`, or `BELL / (Z,NZ)`. Counter symbols are
Z (zero), NZ (non-zero), and - (ignored) EXCLUSIVELY!!, don't invent conditions. Use the exact section headings `Initial
State`, `Initial Counters`, `State Transitions`, `Counter Transitions`, and `Reward
Transitions`. Each transition must be `EXPRESSION -> VALUE`; state values are integer
states (or `-1` for terminal), counter values (-1, 0, 1) are integer tuples that decrement, increment or maintain counters, and reward values are
numbers. Within each State, use the same transition expressions in all three transition
sections.

ALWAYS negatively reward ordinary movement 

HIGHLY penalize picking up collectables (keys, coins) prematurely if they're available in the current environment, and terminate the episode

DO NOT DO THIS:
COIN / (-) -> (c-1,)
DO THIS:
COIN / (-) -> (-1,)

Do not include counter variables in the value section.

Use counters only when needed to track repetition. Counter updates are additive. Keep counters unchanged on ordinary movement.
For simple ordered sequences, use states without counters

For a stage requiring N repetitions, use this pattern:
- Initialize its counter to N - 1.
- While that counter is NZ, consume the event, decrement it, and stay
  in the current stage.
- When that counter is Z, consume the final required event and advance
  to the next stage

Counter conditions are positional. With c0 = (bell_count, coin_count), use (NZ,-) / (Z,-)
for bells and (-,NZ) / (-,Z) for coins. A single (Z) or (NZ) always checks the first counter, regardless of the current state.
"""

COMMON_PITFALLS = """
Common Pitfalls:

MAKE SURE DESIRED_TRACE PASSES WHEN PUT THROUGH CRM

Use `EVENT_EXPRESSION / COUNTER_CONDITION`, such as `RED_KEY / (-)`, `COIN / (NZ)`,
`GOAL / (-)`, or `BELL / (Z,NZ)`. Counter symbols are
Z (zero), NZ (non-zero), and - (ignored) EXCLUSIVELY!!, don't invent conditions. Use the exact section headings `Initial
State`, `Initial Counters`, `State Transitions`, `Counter Transitions`, and `Reward
Transitions`. Each transition must be `EXPRESSION -> VALUE`; state values are integer
states (or `-1` for terminal), counter values (-1, 0, 1) are integer tuples that decrement, increment or maintain counters, and reward values are
numbers. Within each State, use the same transition expressions in all three transition
sections.

DO NOT DO THIS:
COIN / (-) -> (c-1,)
DO THIS:
COIN / (-) -> (-1,)

Do not include counter variables in the value section.

Use counters only when needed to track repetition. Counter updates are additive. Keep counters unchanged on ordinary movement.
For simple ordered sequences, use states without counters

For a stage requiring N repetitions, use this pattern:
- Initialize its counter to N - 1.
- While that counter is NZ, consume the event, decrement it, and stay
  in the current stage.
- When that counter is Z, consume the final required event and advance
  to the next stage

Counter conditions are positional. With c0 = (bell_count, coin_count), use (NZ,-) / (Z,-)
for bells and (-,NZ) / (-,Z) for coins. A single (Z) or (NZ) always checks the first counter, regardless of the current state.
"""

TRAIN_RESULTS = """The following are results from an agent's performance from training using the CRM:"""

FEEDBACK_FORMAT = """
Using the CRM, its desired trace, the actual agent trace, and the trained agent's performance metrics, give
constructive feedback for a generative LLM to improve the CRM. Include the task,
training results (episode reward, maximum reward and efficiency),
failure analysis, and a suggested revision.
"""

TASK_PROMPT = """
========================================
          DEFINE YOUR TASK
========================================

You are controlling an agent in a 10 x 10 grid-world environment. Describe WHAT the
agent should accomplish; do not design the map or specify object locations. A suitable
environment will be generated automatically.

Available objects: collectible coloured keys (yellow, red, green, blue), countable coins, bells, goals/exits,
hazards, and impassable walls. The agent can move up, down, left, right, and interact
with objects such as bells.

Tasks can involve objectives, counts, bells, ordering, alternatives, specifying the presence of
hazards, or combinations. Example: "Collect the red key, then collect at least three
coins, ring the bell twice, and finally reach the exit without touching hazards."

Please enter the task you want the agent to accomplish:

> """

# Concise variants for prompt-length A/B testing. Required section markers and
# grammar are retained because the application parses the model response.
GENERATOR_CONTEXT_SHORT = """
Generate CRM task logic for a 10x10 GridWorld, never layout or paths. Events:
RED_KEY, BLUE_KEY, GREEN_KEY, YELLOW_KEY, COIN, BELL, GOAL, HAZARD.
Actions: UP/DOWN/LEFT/RIGHT/INTERACT. Preserve requested order/counts;
use counters only for counting. Terminate on final GOAL.
"""

CRM_GENERATION_PROMPT_SHORT = """
Reject invalid/unrelated tasks with only <<<INVALID_TASK>>>.
Encode the requested event order, then GOAL. Events: RED_KEY, BLUE_KEY,
GREEN_KEY, YELLOW_KEY, COIN, BELL, GOAL, HAZARD.
Initial sections: DESIRED_TRACE, ENVIRONMENT, CRM.
Trace syntax: EVENT -> EVENT x COUNT -> GOAL; counts are positive integers.
For ordered single events use successive states, c0 = (), and no counters.
GREEN_KEY -> BLUE_KEY -> RED_KEY -> GOAL uses states 0 -> 1 -> 2 -> 3 -> -1.
Counting example COIN x 2 -> GREEN_KEY -> BELL x 3 -> GOAL: c0=(2,3).
State 0: COIN/(NZ,-)->0, delta=(-1,0); GREEN_KEY/(Z,-)->1, delta=(0,0).
State 1: BELL/(-,NZ)->1, delta=(0,-1); GOAL/(-,Z)->-1, delta=(0,0).
After counting, consume NEXT_EVENT in the counting state, not a separate state
or an extra counted event. Match condition/update length to c0.
Only final GOAL enters -1 directly; no extra state after GOAL.
Refinements output only <<<CRM>>>; preserve trace and environment. No Python.
"""

ENVIRONMENT_CONTEXT_SHORT = """
ENVIRONMENT: minimum object counts; enough coins/keys for the entire trace.
One bell supports repeated interactions. Hazards are Boolean. No coordinates
or walls; Python generates layout.
"""

ENVIRONMENT_SEMANTICS_SHORT = """
Entering a key/coin consumes it and emits its event. INTERACT on a bell emits
BELL repeatedly. Entering goal/hazard emits GOAL/HAZARD. Ordinary movement
emits no event; walls block movement. CRM encodes order, not paths.
"""

REFINEMENT_CONTEXT_SHORT = """
Repair event order using feedback. Keep trace/environment fixed and correct
prefixes. Advance one stage per required event; GOAL -> -1 only at the end.
Output only <<<CRM>>>.
"""

GENERATION_FORMAT_SHORT = """
Use exactly these sections and syntax:

<<<DESIRED_TRACE>>>
RED_KEY x 2 -> GOAL

<<<ENVIRONMENT>>>
{"coins": 0, "keys": {"red": 2, "blue": 0, "green": 0, "yellow": 0}, "bells": 0, "hazards": false}

<<<CRM>>>
Initial State
u0 = 1
Initial Counters
c0 = ()
State Transitions
State 0
    GOAL / (-) -> -1
    / (-) -> 0
Counter Transitions
State 0
    GOAL / (-) -> ()
    / (-) -> ()
Reward Transitions
State 0
    GOAL / (-) -> 10.0
    / (-) -> -0.01

"""

REFLECTION_FORMAT_SHORT = """
Copy supplied trace and JSON exactly; diagnose event order:
<<<AGENT_TRACE>>>
["RED_KEY", "GOAL"]
<<<RUN_INFORMATION>>>
{...}
<<<REFLECTION>>>
...
"""

FAIL_PARSE_SHORT = """Fix CRM syntax."""

CRM_FORMAT_SHORT = """
Example headings; match states/expressions across sections.
Rows: EVENT / CONDITION -> VALUE; and/or/not allowed.
c0 = () without counts. Counter updates are additive tuples; initialise counts,
decrement events. Conditions test BEFORE updates: Z=zero, NZ=nonzero;
(-,NZ,-) tests only counter 2. Put / (-) default self-loops last.
State targets are integers; final GOAL -> -1 directly. Rewards: numbers.
Single-counter tuples need commas: (2,), (-1,), (0,).
"""

COMMON_PITFALLS_SHORT = COMMON_PITFALLS

TRAIN_RESULTS_SHORT = """Trained-agent results:"""

FEEDBACK_FORMAT_SHORT = """
Identify incorrect transitions using desired/actual traces and metrics;
suggest precise CRM edits.
"""

TASK_PROMPT_SHORT = """
========================================
          DEFINE YOUR TASK
========================================

You are controlling an agent in a 10 x 10 grid-world environment. Describe WHAT the
agent should accomplish; do not design the map or specify object locations. A suitable
environment will be generated automatically.

Available objects: collectible coloured keys (yellow, red, green, blue), countable coins, bells, goals/exits,
hazards, and impassable walls. The agent can move up, down, left, right, and interact
with objects such as bells.

Tasks can involve objectives, counts, bells, ordering, alternatives, specifying the presence of
hazards, or combinations. Example: "Collect the red key, then collect at least three
coins, ring the bell twice, and finally reach the exit without touching hazards."

Please enter the task you want the agent to accomplish:

> """

_PROMPT_SPECIFICATION_NAMES = (
    "GENERATOR_CONTEXT", "CRM_GENERATION_PROMPT", "ENVIRONMENT_CONTEXT",
    "ENVIRONMENT_SEMANTICS", "REFINEMENT_CONTEXT", "GENERATION_FORMAT",
    "REFLECTION_FORMAT", "FAIL_PARSE", "CRM_FORMAT", "COMMON_PITFALLS", "TRAIN_RESULTS",
    "FEEDBACK_FORMAT", "TASK_PROMPT",
)


def select_specifications(long=True):
    """Return the complete long or concise prompt specification set."""
    suffix = "" if long else "_SHORT"
    return {name: globals()[name + suffix] for name in _PROMPT_SPECIFICATION_NAMES}

# Backwards-compatible names for existing callers.
Generator_Context = GENERATOR_CONTEXT
crm_generation_prompt = CRM_GENERATION_PROMPT
environment_context = ENVIRONMENT_CONTEXT
environment_semantics = ENVIRONMENT_SEMANTICS
refinement_context = REFINEMENT_CONTEXT
generation_format = GENERATION_FORMAT
reflection_format = REFLECTION_FORMAT
fail_parse = FAIL_PARSE
crm_format = CRM_FORMAT
common_pitfalls = COMMON_PITFALLS
train_results = TRAIN_RESULTS
feedback_format = FEEDBACK_FORMAT
task_prompt = TASK_PROMPT
Generator_Context_short = GENERATOR_CONTEXT_SHORT
crm_generation_prompt_short = CRM_GENERATION_PROMPT_SHORT
environment_context_short = ENVIRONMENT_CONTEXT_SHORT
environment_semantics_short = ENVIRONMENT_SEMANTICS_SHORT
refinement_context_short = REFINEMENT_CONTEXT_SHORT
generation_format_short = GENERATION_FORMAT_SHORT
reflection_format_short = REFLECTION_FORMAT_SHORT
fail_parse_short = FAIL_PARSE_SHORT
crm_format_short = CRM_FORMAT_SHORT
train_results_short = TRAIN_RESULTS_SHORT
feedback_format_short = FEEDBACK_FORMAT_SHORT
task_prompt_short = TASK_PROMPT_SHORT

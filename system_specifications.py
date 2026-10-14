"""Prompt and format specifications used by the CRM generation loop."""

CRM_GENERATION_PROMPT = """
## CRM Generation Instructions

You are generating a **Counting Reward Machine (CRM)** for a fixed 10x10 GridWorld.
The agent can move UP, DOWN, LEFT, RIGHT, and INTERACT. The environment contains
collectible keys and coins, bells, a goal, hazards, and impassable walls.
The CRM describes the desired event trace and its temporal relationships only,
never object positions, paths, layout, or movement. Do not invent, remove, or reorder
events required by the user's task. Complete the task, then move to the GOAL.

Accept achievable objectives using the supported events, including a single event.
"Pick up" means collect; "a" or "one" means one object. An omitted goal is implicit.
For example, "Pick up a coin" is valid and requires COIN -> GOAL.
Reject only unrelated, unsupported, contradictory, or unachievable objectives with
only <<<INVALID_TASK>>>, if a task is ambiguous but doable, just select one
interpretation and continue. Brevity or an omitted explicit goal is not a reason to reject.

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
the minimum number of objects needed to make fully fulfill the task specification, describing WHAT must
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
Bells are reusable: one bell supports any number of rings. 
Bell count means physical objects, not interactions.
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
constructive feedback for a generative LLM to improve the CRM, try finding errors or
flaws in the CRM logic. Include the task,
training results (episode reward, maximum reward and efficiency),
failure analysis, and a suggested revision.
"""

TASK_PROMPT = """
========================================
          DEFINE YOUR TASK
========================================

Describe a clear, achievable task for an agent in a generated 10 x 10 GridWorld.
The task must be expressible using these events:
- Collect a red, blue, green, or yellow key.
- Collect coins.
- Ring a bell.
- Reach the goal/exit.
- Specify the presence of hazards.

Specify the required order and any counts clearly. Use positive whole numbers
for counts; exact counts give the clearest target sequence. Every task finishes
at the goal/exit after the other requirements, even if you omit that final step.
Keys and coins are collected once per object. Bells are reusable, so one bell
can be rung repeatedly. You may ask for hazards to be present and avoided.
Keep the required physical objects small enough to fit in the 10 x 10 grid.

Describe objectives only. Object positions, walls, the starting position, and
routes are generated automatically. Do not specify coordinates, map layouts,
movement commands, or unsupported actions/objects (such as fighting enemies,
opening doors, carrying items to another character, or collecting apples).
Unrelated text and contradictory requirements are invalid.
For a predictable test, give one ordered sequence rather than multiple alternatives.

Valid examples:
- "Reach the goal."
- "Collect the red key, collect 3 coins, ring the bell twice, then reach the goal."
- "Ring the bell 10 times, then reach the goal without touching hazards."

Please enter the task you want the agent to accomplish:

> """

# Concise variants target roughly 40% of normal specification tokens.
# Preserve parsed markers and grammar; the task-entry prompt stays identical.
CRM_GENERATION_PROMPT_SHORT = """
10x10 GridWorld CRM. Actions: UP/DOWN/LEFT/RIGHT/INTERACT.
Task event logic only; no layout/paths/movement. Preserve requested events/order.
Events:
RED_KEY, BLUE_KEY, GREEN_KEY, YELLOW_KEY, COIN, BELL, GOAL, HAZARD.
u0, c0, state/counter/reward transitions required.
Rows: EVENT_EXPRESSION / COUNTER_CONDITION -> VALUE. Boolean and/or/not allowed.
Conditions: -=ignore, Z=zero, NZ=nonzero; positional tuples.
Counters only for repetition; ordering: states, c0=().
First output <<<DESIRED_TRACE>>>, <<<ENVIRONMENT>>>, <<<CRM>>>.
Trace: EVENT -> EVENT x COUNT -> GOAL; positive counts, default one.
One final GOAL; no movement/no-ops. Refine: only <<<CRM>>>;
trace/environment fixed. No Python.
Accept supported single-event tasks. "Pick up"=collect; "a"/"one"=1; omitted goal is implicit.
"Pick up a coin" is valid: COIN -> GOAL.
Reject only unrelated/unsupported/contradictory/unachievable objectives: <<<INVALID_TASK>>>.
"""

ENVIRONMENT_CONTEXT_SHORT = """
Initial ENVIRONMENT: ABSOLUTE MINIMUM counts of coins, red/blue/green/yellow keys, bells,
for the entire trace. No coordinates/walls; Python chooses layout and agent start.
"""

ENVIRONMENT_SEMANTICS_SHORT = """
10x10 grid; UP/DOWN/LEFT/RIGHT move one cell. Enter keys/coins to collect and
emit their events; goal/hazard entry emits GOAL/HAZARD. INTERACT on bell emits
BELL. Walls block without events. Python chooses layout; CRM encodes event order.
"""

REFINEMENT_CONTEXT_SHORT = """
Use trace comparison, run data, and reflection to repair CRM order.
Keep trace/environment fixed and unrepeated; preserve correct prefixes.
Output only <<<CRM>>>.
"""

GENERATION_FORMAT_SHORT = """
Adapt this format:

<<<DESIRED_TRACE>>>
GOAL

<<<ENVIRONMENT>>>
{"coins":0,"keys":{"red":0,"blue":0,"green":0,"yellow":0},"bells":0,"hazards":false}

<<<CRM>>>
Initial State
u0 = 0
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
Copy trace/JSON exactly; explain desired-trace differences:
<<<AGENT_TRACE>>>
["GOAL"]
<<<RUN_INFORMATION>>>
{...raw JSON...}
<<<REFLECTION>>>
...
"""

FAIL_PARSE_SHORT = """Fix CRM parse error."""

CRM_FORMAT_SHORT = """
Headings: Initial State, Initial Counters, State Transitions,
Counter Transitions, Reward Transitions. Define u0 and c0.
Rows: EVENT_EXPRESSION / COUNTER_CONDITION -> VALUE.
Match expressions across sections. States: integers, -1 terminal; rewards: numbers.
Counter deltas: additive tuples of -1/0/1, e.g. (-1,), never (c-1,).
Movement: negative reward, unchanged counters. Premature keys/coins: heavy penalty
and termination. Ordering uses states; repetition uses counters.
N repetitions: initialize N-1; NZ consumes/decrements/stays, Z consumes final
event/advances. Test before updating. Only Z=zero, NZ=nonzero, -=ignore.
Conditions are positional: (NZ,-) tests counter 1, (-,NZ) counter 2;
lone Z/NZ always tests counter 1. Match tuple lengths to c0.
"""

COMMON_PITFALLS_SHORT = """
Ensure CRM accepts the exact desired trace. Headings: Initial State, Initial Counters, State Transitions,
Counter Transitions, Reward Transitions.
Rows: EVENT / CONDITION -> VALUE; expressions match across sections.
States integer (-1 terminal), rewards numeric, counter deltas additive -1/0/1
tuples: (-1,), never (c-1,). Movement preserves counters. Ordering uses states;
repetition uses counters. Initialize N-1; NZ consumes/decrements/stays,
Z consumes final event/advances. Only Z/NZ/- conditions; positions matter:
(NZ,-) tests counter 1, (-,NZ) counter 2, lone Z/NZ counter 1.
Match conditions/updates to c0 length; use full supported event names.
"""

TRAIN_RESULTS_SHORT = """CRM-trained agent results:"""

FEEDBACK_FORMAT_SHORT = """
Constructive CRM feedback: task, desired/actual traces, training reward/maximum/
efficiency, failure analysis, suggested revision.
"""

TASK_PROMPT_SHORT = TASK_PROMPT

_PROMPT_SPECIFICATION_NAMES = (
    "CRM_GENERATION_PROMPT", "ENVIRONMENT_CONTEXT",
    "ENVIRONMENT_SEMANTICS", "REFINEMENT_CONTEXT", "GENERATION_FORMAT",
    "REFLECTION_FORMAT", "FAIL_PARSE", "CRM_FORMAT", "COMMON_PITFALLS", "TRAIN_RESULTS",
    "FEEDBACK_FORMAT", "TASK_PROMPT",
)


def select_specifications(long=True):
    """Return the complete long or concise prompt specification set."""
    suffix = "" if long else "_SHORT"
    return {name: globals()[name + suffix] for name in _PROMPT_SPECIFICATION_NAMES}

# Backwards-compatible names for existing callers.
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

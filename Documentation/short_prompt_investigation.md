# Three-key CRM failure and short prompt changes

Task: Pick up a green key, then a blue key, then a red key.
Generator: `gemini-3.5-flash-lite`; short specifications were selected.

The saved final correction, trace `tr-fd4c99c2165731c9aeb3072f8299003b`,
initialised three counters `(1, 1, 1)` and used `(NZ)` for every key.
This condition tests only counter 1. GREEN_KEY moved State 0 to State 1
and changed the counters to `(0, 1, 1)`. BLUE_KEY and RED_KEY then took
self-loops because their conditions tested that now-zero first counter.
GOAL could not terminate, so all three repeated corrections failed.
The final stage also routed GOAL through State 4 before a later terminal
transition, which would require an additional step even with correct guards.

The earlier generation passed the previous validator despite the same counter
stall: GOAL could terminate from State 1. It therefore allowed completion after
only the green key. Validation previously checked full-trace termination without
checking GOAL at incomplete prefixes.

Changes:

- Single-event ordering uses successive states and no counters.
- Short instructions explain positional conditions, additive updates, tuple
  commas, and direct final-GOAL termination.
- A compact mixed counting example consumes the next event from the counting
  state once its remaining counter is zero. It requires no extra counted event.
- Validation rejects premature GOAL termination and explains replay failures.
- Parsing rejects counter updates of the wrong length and conditions that refer
  beyond the initial counter tuple.
- Corrections receive the fixed trace, previous CRM, actual error, syntax example,
  and generation rules. Alignment failures are recorded in MLflow.

Final Gemini token counts:

| Specification input | Short | Regular | Short / regular |
| --- | ---: | ---: | ---: |
| Three-key initial prompt | 692 | 1,479 | 46.8% |
| Complete specification set | 913 | 2,147 | 42.5% |

Independent cl100k_base and o200k_base checks also kept initial, judge,
and refinement specification inputs below half their regular equivalents.
Task data, model outputs, and accumulated conversation history are separate.

Live three-key comparison: the final worked-example revision passed CRM
validation on 3/3 fresh generations; the original saved short prompt passed
0/3 in the same comparison. An earlier comparison gave 3/3 versus 1/3.
These are small stochastic samples, not measured task success rates.

Two coins, a green key, and three bell rings passed generation validation.
The larger task of 23 bell rings, four coins, and a red key initially produced
invalid counter tuples. Concrete correction feedback repaired it on correction
3. Complex counting remains less reliable than simple ordering.

Validation replayed event sequences and checked CRM parsing. It did not rerun
agent training. Raw samples and token counts are saved under `outputs/short_prompt*`.

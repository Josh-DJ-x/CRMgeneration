<<<DESIRED_TRACE>>>

["COIN", "COIN", "COIN", "COIN", "COIN", "RED_KEY", "GOAL"]

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
c0 = (0,)

State Transitions

State 0
    COIN / (-)            -> 0
    RED_KEY / (NZ)        -> 1
    / (-)                 -> 0

State 1
    GOAL / (-)            -> -1
    / (-)                 -> 1

Counter Transitions

State 0
    COIN / (-)            -> (1,)
    RED_KEY / (NZ)        -> (0,)
    / (-)                 -> (0,)

State 1
    GOAL / (-)            -> (0,)
    / (-)                 -> (0,)

Reward Transitions

State 0
    COIN / (-)            -> 0.1
    RED_KEY / (NZ)        -> 1.0
    / (-)                 -> -0.01

State 1
    GOAL / (-)            -> 10.0
    / (-)                 -> -0.01

The desired trace is the exact ordered sequence of task events required for
successful completion. It must be a JSON array containing only the supported event
names. Do not include ordinary movement steps or actions that emit no event.

`<<<DESIRED_TRACE>>>` and `<<<ENVIRONMENT>>>` are emitted only in the first
generation. Refinements must contain only `<<<CRM>>>`; they use the stored initial
trace and environment specification unchanged.

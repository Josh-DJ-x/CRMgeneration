import os
from google import genai
from openai import OpenAI
import ast
import re
import json
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum, auto

from pycrm.crossproduct import CrossProduct
from pycrm.label import LabellingFunction
from pycrm.automaton import CountingRewardMachine

from collections import defaultdict

class LLM(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class Gemini(LLM):

    def __init__(self, api_key):

        self.client = genai.Client(api_key=api_key)

        self.chat = self.client.chats.create(
            model="gemini-3.1-pro-preview"
        )


    def generate(self, prompt: str) -> str:

        response = self.chat.send_message(prompt)

        return response.text

class OpenRouter(LLM):

    def __init__(self, api_key, model="openrouter/free"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        self.model = model
        self.messages = []

    def generate(self, prompt: str) -> str:

        self.messages.append({
            "role": "user",
            "content": prompt
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )

        content = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": content
        })

        return content

    def reset_history(self):
        self.messages = []

def extraction(response: str, initial: bool = False):
    """
    Extract REQUIREMENTS, ENVIRONMENT and CRM sections from an LLM response.
    """

    pattern = (
        r"<<<REQUIREMENTS>>>\s*(.*?)"
        r"(?:<<<ENVIRONMENT>>>\s*(\{.*?\}))?"
        r"\s*<<<CRM>>>\s*(.*)"
    )

    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if match is None:
        raise ValueError("LLM response could not be parsed.")

    requirements = match.group(1).strip()
    environment = match.group(2)
    crm = match.group(3).strip()

    if initial:
        if environment is None:
            raise ValueError("Initial generation did not contain ENVIRONMENT.")
        return requirements, crm, environment.strip()

    return requirements, crm, None

class Event(Enum):
    RED_KEY = auto()
    BLUE_KEY = auto()
    GREEN_KEY = auto()
    YELLOW_KEY = auto()

    COIN = auto()
    BELL = auto()
    GOAL = auto()
    HAZARD = auto()

GRID_OBJECTS = {
    "R": Event.RED_KEY,
    "B": Event.BLUE_KEY,
    "E": Event.GREEN_KEY,
    "Y": Event.YELLOW_KEY,
    "C": Event.COIN,
    "S": Event.BELL,
    "G": Event.GOAL,
    "H": Event.HAZARD,
}

class GridWorld(gym.Env):

    def __init__(self, grid):
        super().__init__()

        self.grid = np.array(grid, dtype=object)

        self.height = len(grid)
        self.width = len(grid[0])

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=max(self.height, self.width) - 1,
            shape=(2,),
            dtype=np.int32
        )

        self.agent_position = None

    def find_agent(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row][col] == "A":
                    self.grid[row][col] = "."
                    return (row, col)

        raise ValueError("Grid does not contain an agent.")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_position = self.find_agent()
        self.steps = 0

        return self.get_observation(), {}

    def get_observation(self):
        return np.array(
            self.agent_position,
            dtype=np.int32
        )

    def step(self, action):

        self.move_agent(action)

        observation = self.get_observation()

        terminated = self.is_terminal()
        truncated = False
        self.steps += 1

        truncated = self.steps >= 200

        return observation, 0.0, terminated, truncated, {}

    def move_agent(self, action):

        row, col = self.agent_position

        new_row = row
        new_col = col

        if action == 0:       # UP
            new_row -= 1
        elif action == 1:     # DOWN
            new_row += 1
        elif action == 2:     # LEFT
            new_col -= 1
        elif action == 3:     # RIGHT
            new_col += 1
        elif action == 4:     # INTERACT
            return

        if not (0 <= new_row < self.height):
            return

        if not (0 <= new_col < self.width):
            return

        if self.grid[new_row][new_col] == "#":
            return

        self.agent_position = (new_row, new_col)

        symbol = self.grid[new_row][new_col]

        if symbol in {"R", "B", "E", "Y", "C"}:
            self.grid[new_row][new_col] = "."

    def is_terminal(self):
        row, col = self.agent_position

        return self.grid[row][col] == "G"

    def render(self):

        display = self.grid.copy()

        row, col = self.agent_position
        display[row][col] = "A"

        for row in display:
            print(" ".join(row))

        print()


def generate_gridworld(seed, environment_json):

    spec = parse_environment(environment_json)

    rng = np.random.default_rng(seed)

    grid = [["." for _ in range(10)] for _ in range(10)]

    # Place agent.
    grid[0][0] = "A"

    # Place goal.
    grid[9][9] = "G"

    ...
    # Place objects procedurally.

    return GridWorld(grid)

def parse_environment(environment_text):
    try:
        environment = json.loads(environment_text)
    except json.JSONDecodeError:
        raise ValueError("Environment specification is not valid JSON.")

    if not validate_environment_spec(environment):
        raise ValueError("Invalid environment specification.")

    return environment

def validate_environment_spec(environment):

    required = {
        "coins",
        "keys",
        "goals",
        "bells",
        "hazards"
    }

    if set(environment.keys()) != required:
        return False

    if not isinstance(environment["coins"], int):
        return False

    if not isinstance(environment["goals"], int):
        return False

    if not isinstance(environment["bells"], int):
        return False

    if not isinstance(environment["hazards"], int):
        return False

    if any(
        not isinstance(v, int) or v < 0
        for v in environment["keys"].values()
    ):
        return False

    valid_colours = {"red", "blue", "green", "yellow"}

    if set(environment["keys"].keys()) != valid_colours:
        return False

    return True

class GridWorldLabellingFunction(LabellingFunction[np.ndarray, int]):

    def __init__(self, gridworld):
        self.gridworld = gridworld

    @LabellingFunction.event
    def detect_event(self, obs, action, next_obs):

        prev_row, prev_col = obs
        next_row, next_col = next_obs

        if (prev_row, prev_col) == (next_row, next_col):
            return None

        symbol = self.gridworld.grid[next_row][next_col]

        return GRID_OBJECTS.get(symbol)

class GridWorldCrossProduct(CrossProduct):

    def __init__(self, ground_env, crm, lf, max_steps):

        super().__init__(
            ground_env,
            crm,
            lf,
            max_steps
        )

class TestCRM(CountingRewardMachine):

    def __init__(self):
        super().__init__(env_prop_enum=Event)

    @property
    def u_0(self):
        return 0

    @property
    def c_0(self):
        return (0,)

    def _get_state_transition_function(self):
        return {
            0: {
                "RED_KEY / (-)": 1,
                "/ (-)": 0,
            },

            1: {
                "GOAL / (-)": -1,
                "/ (-)": 1,
            }
        }

    def _get_counter_transition_function(self):
        return {
            0: {
                "RED_KEY / (-)": (0,),
                "/ (-)": (0,),
            },

            1: {
                "GOAL / (-)": (0,),
                "/ (-)": (0,),
            }
        }

    def _get_reward_transition_function(self):
        return {
            0: {
                "RED_KEY / (-)": 1,
                "/ (-)": -0.01,
            },

            1: {
                "GOAL / (-)": 10,
                "/ (-)": -0.01,
            }
        }

    def sample_counter_configurations(self):
        return [(0,)]

def train_agent(
    env,
    episodes=5000,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=0.1
):

    q_table = defaultdict(
        lambda: np.zeros(env.action_space.n)
    )

    episode_returns = []

    for episode in range(episodes):

        obs, _ = env.reset()

        state = tuple(obs)

        episode_return = 0.0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            # Epsilon-greedy action selection
            if (
                np.random.random() < epsilon
                or np.all(q_table[state] == 0)
            ):
                action = env.action_space.sample()

            else:
                action = int(
                    np.argmax(q_table[state])
                )

            next_obs, reward, terminated, truncated, _ = env.step(
                action
            )

            next_state = tuple(next_obs)

            episode_return += reward

            # Q-learning update
            if terminated or truncated:

                target = reward

            else:

                target = (
                    reward
                    + discount_factor
                    * np.max(q_table[next_state])
                )

            td_error = (
                target
                - q_table[state][action]
            )

            q_table[state][action] += (
                learning_rate
                * td_error
            )

            state = next_state

        episode_returns.append(episode_return)

    return q_table, episode_returns

def crm_parse(crm_text: str):

    crm = {
        "u0": None,
        "c0": None,
        "state_transitions": {},
        "counter_transitions": {},
        "reward_transitions": {}
    }

    # Initial state
    crm["u0"] = int(
        re.search(r"u0\s*=\s*(\d+)", crm_text).group(1)
    )

    # Counters
    crm["c0"] = ast.literal_eval(
        re.search(r"c0\s*=\s*(\([^)]+\))", crm_text).group(1)
    )

    current_section = None
    current_state = None

    for line in crm_text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("State Transitions"):
            current_section = "state"
            continue

        if line.startswith("Counter Transitions"):
            current_section = "counter"
            continue

        if line.startswith("Reward Transitions"):
            current_section = "reward"
            continue

        if line.startswith("State "):
            current_state = int(line.split()[1])

            crm[current_section + "_transitions"][current_state] = {}
            continue

        if "->" in line:

            left, right = map(str.strip, line.split("->"))

            if current_section == "state":
                crm["state_transitions"][current_state][left] = int(right)

            elif current_section == "counter":
                crm["counter_transitions"][current_state][left] = ast.literal_eval(right)

            elif current_section == "reward":
                crm["reward_transitions"][current_state][left] = float(right)

    return crm
    
#Returns true if all requirements have been met
def requirements_met(reflection):
    """
    Returns True if all requirements are satisfied.
    """

    match = re.search(
        r"<<<VERIFICATION>>>\s*(\{.*\})",
        reflection,
        re.DOTALL
    )

    if match is None:
        raise SystemExit("Error: Verification JSON not found.")

    verification = json.loads(match.group(1))

    return verification["all_requirements_met"]

Generator_Context = """
You are generating a Counting Reward Machine (CRM) for a fixed
10x10 GridWorld environment.

The agent can move:
- UP
- DOWN
- LEFT
- RIGHT
- INTERACT

The environment contains:

- RED_KEY / BLUE_KEY / GREEN_KEY / YELLOW_KEY:
  Collectible keys. A key is collected when the agent enters its cell.

- COIN:
  A collectible object. Each coin collected increments the coin counter.

- BELL:
  A bell that can be rung using INTERACT.

- GOAL:
  The task destination.

- HAZARD:
  Entering a hazard cell produces a HAZARD event and constitutes
  failure if the task specifies that hazards must be avoided.

- WALL:
  Impassable. Walls do not produce task events.

The CRM should represent ONLY the task requirements and their temporal
relationships. Do not attempt to specify the physical map, object
coordinates, movement paths, or environment layout.

The following event labels are available:

RED_KEY
BLUE_KEY
GREEN_KEY
YELLOW_KEY
COIN
BELL
GOAL
HAZARD

The CRM may use counters when the task requires counting.

For example:

"Collect at least 5 coins before reaching the goal."

requires the CRM to track the number of COIN events and only permit
successful completion after the counter reaches 5.

For a requirement such as:

"Collect the red key while having at least 5 coins."

the CRM must ensure that RED_KEY occurs only when the coin counter
is at least 5.

Generate the CRM from the user's task description.
Do not invent requirements that are not present in the task.
Do not remove requirements that are present in the task.
"""

crm_generation_prompt = """
## CRM Generation Instructions

You are generating a **Counting Reward Machine (CRM)** for a GridWorld task.

The CRM describes **task logic only**. It does **not** describe the physical environment, object positions, paths, or movement.

### GridWorld Event Vocabulary

The environment produces the following symbolic events:

* `RED_KEY`
* `BLUE_KEY`
* `GREEN_KEY`
* `YELLOW_KEY`
* `COIN`
* `BELL`
* `HAZARD`

These events are generated automatically by the GridWorld labelling function.

### CRM Components

Every CRM must contain:

1. **Initial State**

   * A single starting state (`u0`).

2. **Initial Counters**

   * One or more counters (`c0`) starting at integer values.

3. **State Transition Function**

   * Specifies which machine state to transition to when an event expression is satisfied.

4. **Counter Transition Function**

   * Specifies how counters increment, decrement, or remain unchanged for each transition.

5. **Reward Transition Function**

   * Specifies the reward assigned to each transition.

### Transition Format

Every transition must use:

```text
EVENT_EXPRESSION / COUNTER_CONDITION
```

Examples:

```text
RED_KEY / (-)
GOAL / (-)
COIN / (-)
RED_KEY and not HAZARD / (-)
GOAL and not HAZARD / (NZ)
```

### Event Expressions

Use Boolean logic:

* `and`
* `or`
* `not`
* parentheses `(...)`

Examples:

* `RED_KEY and COIN`
* `RED_KEY or BLUE_KEY`
* `GOAL and not HAZARD`

### Counter Conditions

Use:

* `(-)` Ignore counters.
* `(Z)` Counter is zero.
* `(NZ)` Counter is non-zero.
* `(Z,NZ)` First counter is zero, second counter is non-zero.

### Counting Rules

Counters are used only when the task requires counting.

Examples:

**Collect five coins before reaching the goal**

* `COIN` increments the counter.
* `GOAL` is only valid once the counter condition is satisfied.

**Collect the red key while having at least five coins**

* The `RED_KEY` transition is only valid when the coin counter satisfies the required condition.

### Output Requirements

Always output three sections:

1. `<<<REQUIREMENTS>>>`
2. `<<<ENVIRONMENT>>>`
3. `<<<CRM>>>`

The CRM must contain:

* Initial State
* Initial Counters
* State Transitions
* Counter Transitions
* Reward Transitions

Do not generate Python code. Generate only the CRM specification.

"""

#This is for the <<<ENVIORNMENT>>> component
environment_context = """
ENVIRONMENT VARIABLE SPECIFICATION

This section applies to the <<ENVIORNMENT>>> part that should appear in your first generation

In your first response only, you must also specify the minimum
number of environment objects required to make the task possible.

You are specifying WHAT objects must exist, not WHERE they exist.

The Python environment generator will determine:
- the positions of all objects
- the positions of walls
- the exact layout of the grid
- the agent's starting position

You must determine:
- how many coins are required
- how many keys of each colour are required
- how many bells are required
- Whether hazards are required

Available key colours:
- red
- blue
- green
- yellow

For counting requirements, provide enough objects to satisfy the
requirement.

For example:

"Collect at least 5 coins before reaching the goal."

requires:

"coins": 5

"Collect 2 red keys and 1 blue key."

requires:

"red": 2
"blue": 1

Do not specify object coordinates or wall locations.
"""

environment_semantics = """
ENVIRONMENT SEMANTICS

The task takes place in a procedurally generated 10×10 GridWorld.

The agent controls a robot that moves one cell at a time.

Movement actions:
- UP
- DOWN
- LEFT
- RIGHT
- INTERACT

Grid symbols:

A = Agent start position.
. = Empty traversable cell.
# = Wall (impassable obstacle).
G = Goal / Exit.
H = Hazard.
C = Coin.
S = Bell.

Keys are represented by coloured symbols:

R = Red key.
B = Blue key.
E = Green key.
Y = Yellow key.

Environment behaviour:

- Entering a key's cell automatically collects that key.
- Entering a coin's cell automatically collects one coin.
- Entering a hazard cell generates a HAZARD event.
- Entering the goal cell generates a GOAL event.
- Standing on a bell and using INTERACT generates a BELL event.
- Walls block movement and generate no events.

The environment generator decides where all objects are placed.
The CRM specifies only the temporal task logic, not object locations.
"""

refinement_context = """
Use the following feedback to improve the CRM. If the feedback indicates that a requirement was not met, update the CRM to ensure that it is satisfied. If the feedback indicates that a requirement was met, do not change the CRM in a way that would violate that requirement.
"""

generation_format = """
IMPORTANT!! - ALL your responses need to be in the following format:

<<<REQUIREMENTS>>>

R1: Collect at least 5 coins.
R2: Collect the red key while having at least 5 coins.
R3: Reach the goal after collecting the red key.

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
  "hazards": 2
}
<<<CRM>>>

Initial State
u0 = 0

Initial Counters
c0 = (0)

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
    / (-)                 -> (0,)

State 1
    / (-)                 -> (0,)

Reward Transitions

State 0
    COIN / (-)            -> 0.1
    / (-)                 -> -0.01

State 1
    GOAL / (-)            -> 10.0
    / (-)                 -> -0.01



The ENVIRONMENT section must only appear in the first generation.
Refinement responses must contain only REQUIREMENTS and CRM.

Do not include coordinates in ENVIRONMENT.
Do not include walls in ENVIRONMENT.
Do not include the agent in ENVIRONMENT.

The Python environment generator will determine the grid layout.

Your response MUST follow this format exactly.
"""

reflection_format = """
You are the judge. It's your responsibility to provide feedback that will later be used by another LLM to improve the provided CRM.
You must determine whether the specified requirements have been satisfied or not as well as provide concise feedback.

IMPORTANT!! - ALL your responses need to be in the following format:

<<<VERIFICATION>>>
{
    "all_requirements_met": true/false,
    "requirements": [
        {
            "id": "R1",
            "status": true/false,
            "reason": ""
        },
        {
            "id": "R2",
            "status": true/false,
            "reason": ""
        },
		...
    ]
}

<<<REFLECTION>>>
...

This is the regular expression (regex) representation for this format:

pattern = (
	r"<<<VERIFICATION>>>\s*(\{.*\})"
	r"<<<REFLECTION>>>\s*(.*)"
)

It's important that your responses are in this EXACT format with no exceptions, as it's necessary for the text to be parsed by my system.
"""

#Crm fails to parse
fail_parse = """
The CRM failed to parse, review the correct CRM syntax to fix this. 
"""

#Explanation of CRM format when chat is initiated with LLM for the first time
crm_format = """
A Counting Reward Machine (CRM) is defined by four components:

1. Initial machine state.
2. Initial counter configuration.
3. State transition function.
4. Counter transition function.
5. Reward transition function.

Events are produced by the GridWorld labelling function.

----------------------------------------
EXAMPLE CRM
----------------------------------------

Task:
Collect the red key, then reach the goal.

Initial State
-------------
u0 = 0

Initial Counters
----------------
c0 = (0)

State Transitions
-----------------

State 0

RED_KEY / (-)        -> State 1
DEFAULT / (-)        -> State 0

State 1

GOAL / (-)           -> Terminal
DEFAULT / (-)        -> State 1

Counter Transitions
-------------------

State 0

RED_KEY / (-)        -> (0)
DEFAULT / (-)        -> (0)

State 1

GOAL / (-)           -> (0)
DEFAULT / (-)        -> (0)

Reward Transitions
------------------

State 0

RED_KEY / (-)        -> +1
DEFAULT / (-)        -> -0.01

State 1

GOAL / (-)           -> +10
DEFAULT / (-)        -> -0.01

----------------------------------------

CRM expressions have the form:

EVENT_EXPRESSION / COUNTER_CONDITION

Examples:

RED_KEY / (-)

COIN / (NZ)

GOAL and not HAZARD / (-)

BELL and RED_KEY / (Z,NZ)

Counter symbols:

Z   = Counter equals zero.
NZ  = Counter is non-zero.
-   = Ignore this counter.
"""

train_results = """
The following are results from an agent's performance from training using the CRM:
"""

#Evaluation
feedback_format = """
Using the Counting Reward Machine (CRM), the list of requirements for the CRM, and the performance metrics of an agent that was trained on this CRM, 
give constructive feedback that a generative LLM could use to improve the CRM. This is the format for <<<REFLECTION>>>

Reflection Format (Example):

Task:
Collect three keys before exiting.

Training Results

Episode Reward:
96

Average Episode Length:
25

Optimal Episode Length:
17

Optimality:
68%   ()

Failure Analysis:

The agent consistently revisits State S2 after collecting the third key.

This indicates that the transition allowing re-entry into S2 is unnecessary.

Suggested Revision:

Prevent transitions from S3 back to S2 once the counter reaches three.
"""

task_prompt = """
========================================
          DEFINE YOUR TASK
========================================

You are controlling an agent in a 10 × 10 grid-world environment.

Describe a task for the agent to accomplish using the objects and
actions available in the environment.

You only need to describe WHAT the agent should do — you do not need
to design the map or specify where objects should be placed. A suitable
environment will be generated automatically for your task.


AVAILABLE OBJECTS
-----------------

• Keys
  Collectible objects that may have different colours/types.

• Coins
  Collectible objects that can be counted.

• Bells
  Objects that can be activated.

• Goals / Exits
  Locations that the agent can reach.

• Hazards
  Dangerous locations or events that can cause a task to fail.

• Walls
  Impassable obstacles.


AVAILABLE ACTIONS
-----------------

The agent can:

• Move up
• Move down
• Move left
• Move right
• Interact with objects such as bells


WHAT YOU CAN DESCRIBE
---------------------

Your task can involve things such as:

• Reaching a particular goal
• Collecting keys or coins
• Collecting a specific number of objects
• Ringing bells
• Completing actions in a particular order
• Completing multiple objectives
• Choosing between alternative objectives
• Avoiding hazards
• Combining several of these requirements


EXAMPLE
-------

"Collect the red key, then collect at least three coins, activate
the bell, and finally reach the exit without touching any hazards."


IMPORTANT
---------

You do NOT need to specify the positions of the walls, keys, coins,
bells, hazards, or exit.

These will be generated automatically to create a valid environment
for your task.

You can specify requirements about the order in which things happen or conditions that must be true when an action occurs.

DESCRIBE YOUR TASK
------------------

Please enter the task you want the agent to accomplish:

> """

print(task_prompt)
    
task_description = input(task_prompt)

gemini = Gemini(
    os.getenv("GEMINI_API_KEY")
)

openrouter = OpenRouter(
    os.getenv("OPENROUTER_API_KEY")
)

# LOOP LOGIC

generation = gemini.generate(
    crm_generation_prompt
    + "\n\n"
    + generation_format
    + "\n\n"
    + environment_context
    + "\n\n"
    + environment_semantics
    + "\n\n"
    + crm_format
    + "\n\n"
    + "Task:\n\n"
    + task_description
)

requirements, crm_text, environment = extraction(generation, initial=1)
environment = parse_environment(environment)

requirements_base = requirements

# Parse CRM
crm = crm_parse(crm_text)

requirements, performance, status = False
iterations = 1

while iterations <= 5:
    parse_fails = 0

    # Procedural CRM check
    while ((crm is None) and parse_fails <= 2):
        parse_fails += 1
        generation = gemini.generate(fail_parse)
        requirements, crm_text = extraction(generation)
        crm = crm_parse(crm_text)

    if parse_fails > 2:
        raise SystemExit("CRM failed to parse too many times.")

    # Setting up Environment
    #grid = generate_gridworld(
    #        seed=42,
    #        environment_spec=environment_spec
    #   )

    #Fixed for now
    grid = [
    ["A", ".", ".", ".", "#", ".", ".", ".", ".", "."],
    [".", ".", "R", ".", "#", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", "#", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "G"],
    ]

    # Setting up pycrm
    ground_env = GridWorld(grid)

    lf = GridWorldLabellingFunction(
        ground_env
    )

    env = GridWorldCrossProduct(
        ground_env=ground_env,
        crm=crm,
        lf=lf,
        max_steps=200
    )

    #Run agent
    obs, info = env.reset(seed=42)

    done = False

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

    # Judge LLM
    if iterations == 1:
        judgement = openrouter.generate(
            "First Generation:\n\n"
            + generation
            + "\n\n"
            + train_results
            + "\n\n"
            + performance
            + "\n\n"
            + reflection_format
            + "\n\n"
            + feedback_format
            + "\n\n"
            + environment_semantics
            + "\n\n"
            + crm_format
            + "\n\n"
            + "Task:\n\n"
            + task_description
        )
    else:
        judgement = openrouter.generate(
            "New Generation\n\n"
            + generation
            + "\n\n"
            + train_results
            + "\n\n"
            + performance
        )

    # Check if requirements have been met
    requirements = requirements_met(judgement)

    # Check if CRM is satisfactory
    if ((crm is not None) and requirements and performance):
        status = True
        break

    # Next generative LLM cycle starts
    generation = gemini.generate(
        refinement_context
        + "\n\n"
        + "Judge Response:\n\n"
        + judgement
        + "\n\n"
        + train_results
        + "\n\n"
        + performance
    )
    requirements, crm_text = extraction(generation)

    crm = crm_parse(crm_text)

    iterations += 1

#Consider doing explicit format checks for all LLM generations

# RESULTS
if status:
    print("Success")
else:
    print("Failure")

print("Requirements:")
print(requirements_base)

print("CRM:")
print(crm)   

#print(gemini.generate("Explain reinforcement learning"))
#print(openrouter.generate("Critique this CRM"))

#openrouter.reset_history()

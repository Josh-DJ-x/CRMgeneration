import os
from google import genai
from openai import OpenAI
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

from collections import deque

from itertools import product

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

        self.start_position = None

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
        self.collected_positions = set()
        self.last_event = None

    def find_agent(self):

        for row in range(self.height):
            for col in range(self.width):

                if self.grid[row][col] == "A":
                    return (row, col)

        raise ValueError("Grid does not contain an agent.")

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self.start_position = self.find_agent()
        self.agent_position = self.start_position

        self.collected_positions = set()

        self.last_event = None

        self.steps = 0

        return self.get_observation(), {}

    def get_observation(self):

        return np.array(
            self.agent_position,
            dtype=np.int32
        )

    def step(self, action):

        reward = -0.01

        self.last_event = None

        self.move_agent(action)

        if action == 4:
            reward = -0.05

        observation = self.get_observation()

        terminated = self.is_terminal()

        self.steps += 1

        truncated = self.steps >= 200

        if self.grid[self.agent_position] == "H":
            reward = -10.0

        return observation, reward, terminated, truncated, {}

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

        # Boundary checking
        if not (0 <= new_row < self.height):
            return

        if not (0 <= new_col < self.width):
            return

        # Wall checking
        if self.grid[new_row][new_col] == "#":
            return

        # Move agent
        self.agent_position = (new_row, new_col)

        symbol = self.grid[new_row][new_col]

        # Detect collectible/event
        if (
            symbol in {"R", "B", "E", "Y", "C"}
            and (new_row, new_col) not in self.collected_positions
        ):

            self.collected_positions.add(
                (new_row, new_col)
            )

            self.last_event = GRID_OBJECTS[symbol]

    def is_terminal(self):
        row, col = self.agent_position
        symbol = self.grid[row][col]

        # Only terminate ground env on hazards.
        # Goal completion is handled by CRM terminal state (-1).
        if symbol == "H":
            return True

        return False

    def render(self):

        display = self.grid.copy()

        start_row, start_col = self.start_position
        display[start_row][start_col] = "."

        for row, col in self.collected_positions:
            if display[row][col] != "G":
                display[row][col] = "."

        row, col = self.agent_position
        display[row][col] = "A"

        for row in display:
            print(" ".join(row))

        print()

def bfs_distances(grid, start):
    """
    Returns a dictionary mapping reachable positions to their shortest distance
    from the start position.
    """

    height = len(grid)
    width = len(grid[0])

    queue = deque([start])
    distances = {start: 0}

    directions = [
        (-1,0),
        (1,0),
        (0,-1),
        (0,1)
    ]

    while queue:

        row, col = queue.popleft()

        for dr, dc in directions:

            nr = row + dr
            nc = col + dc

            if not (0 <= nr < height and 0 <= nc < width):
                continue

            if (nr, nc) in distances:
                continue

            if grid[nr][nc] in {"#", "H"}:
                continue

            distances[(nr, nc)] = distances[(row,col)] + 1
            queue.append((nr,nc))

    return distances

def find_grid_objects(grid):

    objects = {
        "A": None,
        "G": None,
        "coins": [],
        "bells": [],
        "keys": {
            "R": [],
            "B": [],
            "E": [],
            "Y": []
        }
    }

    for row in range(len(grid)):
        for col in range(len(grid[0])):

            symbol = grid[row][col]

            if symbol == "A":
                objects["A"] = (row, col)

            elif symbol == "G":
                objects["G"] = (row, col)

            elif symbol == "C":
                objects["coins"].append((row, col))

            elif symbol == "S":
                objects["bells"].append((row, col))

            elif symbol in objects["keys"]:
                objects["keys"][symbol].append((row, col))

    return objects

def validate_generated_grid(grid, spec):
    """
    Generic validation for a generated environment.

    Checks:
      - Object counts match the specification.
      - Every required object is reachable from A.
      - Goal is reachable from A.

    Returns (valid, metadata).
    """

    objects = find_grid_objects(grid)

    start = objects["A"]
    goal = objects["G"]

    if start is None:
        return False, "Missing agent."

    if goal is None:
        return False, "Missing goal."

    distances = bfs_distances(grid, start)

    # -------------------------------
    # Validate keys
    # -------------------------------
    key_lookup = {
        "red": "R",
        "blue": "B",
        "green": "E",
        "yellow": "Y"
    }

    reachable_required_objects = 0

    for colour, symbol in key_lookup.items():

        expected = spec["keys"][colour]

        actual_positions = objects["keys"][symbol]

        if len(actual_positions) != expected:
            return False, f"Expected {expected} {colour} key(s)."

        for pos in actual_positions:
            if pos not in distances:
                return False, f"{colour.capitalize()} key unreachable."
            reachable_required_objects += 1

    # -------------------------------
    # Validate coins
    # -------------------------------
    if len(objects["coins"]) != spec["coins"]:
        return False, "Incorrect number of coins."

    for coin in objects["coins"]:
        if coin not in distances:
            return False, "Coin unreachable."
        reachable_required_objects += 1

    # -------------------------------
    # Validate bells
    # -------------------------------
    if len(objects["bells"]) != spec["bells"]:
        return False, "Incorrect number of bells."

    for bell in objects["bells"]:
        if bell not in distances:
            return False, "Bell unreachable."
        reachable_required_objects += 1

    # -------------------------------
    # Validate goal
    # -------------------------------
    if goal not in distances:
        return False, "Goal unreachable."

    metadata = {
        "shortest_goal_distance": distances[goal],
        "reachable_required_objects": reachable_required_objects
    }

    return True, metadata

def generate_gridworld(seed, environment_json):

    spec = parse_environment(environment_json)
    rng = np.random.default_rng(seed)

    SIZE = 10

    # Keep generating until we find a valid environment.
    while True:

        # ----------------------------------------------------
        # Fresh empty grid every attempt.
        # ----------------------------------------------------
        
        grid = [["." for _ in range(SIZE)] for _ in range(SIZE)]
        grid[0][0] = "A"
        grid[SIZE-1][SIZE-1] = "G"

        reserved = {(0, 0), (SIZE - 1, SIZE - 1)}

        # ----------------------------------------------------
        # Helper for placing objects.
        # ----------------------------------------------------
        def place_objects(symbol, count):

            placed = 0

            while placed < count:

                row = rng.integers(0, SIZE)
                col = rng.integers(0, SIZE)

                if (row, col) in reserved:
                    continue

                if grid[row][col] != ".":
                    continue

                grid[row][col] = symbol
                placed += 1

        # ----------------------------------------------------
        # Keys
        # ----------------------------------------------------
        key_symbols = {
            "red": "R",
            "blue": "B",
            "green": "E",
            "yellow": "Y",
        }

        for colour, symbol in key_symbols.items():
            place_objects(symbol, spec["keys"][colour])

        # ----------------------------------------------------
        # Coins and bells
        # ----------------------------------------------------
        place_objects("C", spec["coins"])
        place_objects("S", spec["bells"])

        # ----------------------------------------------------
        # Hazards (NEW)
        # ----------------------------------------------------
        if spec["hazards"]:

            # Random but reasonable number.
            hazard_count = rng.integers(3, 8)   # 3–7 hazards.

            place_objects("H", hazard_count)

        # ----------------------------------------------------
        # Validate map.
        # ----------------------------------------------------
        # NEW VALIDATION
        valid, metadata = validate_generated_grid(grid, spec)

        if valid:
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
        "bells",
        "hazards"
    }

    if set(environment.keys()) != required:
        return False

    if not isinstance(environment["coins"], int):
        return False

    if not isinstance(environment["bells"], int):
        return False

    if not isinstance(environment["hazards"], bool):
        return False

    if any(
        not isinstance(v, int) or v < 0
        for v in environment["keys"].values()
    ):
        return False

    valid_colours = {"red", "blue", "green", "yellow"}

    if set(environment["keys"].keys()) != valid_colours:
        return False

    SIZE = 10
    capacity = SIZE * SIZE - 2  # Start and goal occupy two cells.

    total_objects = (
        environment["coins"]
        + environment["bells"]
        + sum(environment["keys"].values())
        + 7   # Maximum hazards if hazards=True
            if environment["hazards"]
            else 0
    )

    if total_objects > capacity:
        return False

    return True

class GridWorldLabellingFunction(LabellingFunction[np.ndarray, int]):

    def __init__(self, gridworld):
        self.gridworld = gridworld

    @LabellingFunction.event
    def detect_event(self, obs, action, next_obs):
        # 1. Check if an item was collected on this step
        if self.gridworld.last_event is not None:
            return self.gridworld.last_event

        # 2. Check current agent position against Goal symbol
        next_row, next_col = next_obs
        if self.gridworld.grid[next_row][next_col] == "G":
            return Event.GOAL

        return None

class GridWorldCrossProduct(CrossProduct):

    def __init__(self, ground_env, crm, lf, max_steps):
        super().__init__(ground_env, crm, lf, max_steps)
        self.action_space = ground_env.action_space

    def get_observation(self):
        ground_obs = self.ground_env.get_observation()      # (row, col)
        crm_obs = self.crm.get_observation()                # one-hot CRM encoding
        return np.concatenate([ground_obs, crm_obs]).astype(np.float32)

class TestCRM(CountingRewardMachine):

    def __init__(self):
        super().__init__(env_prop_enum=Event)

    @property
    def u_0(self) -> int:
        return 0

    @property
    def c_0(self) -> tuple[int, ...]:
        return (4,)

    @property
    def encoded_configuration_size(self) -> int:
        return 2

    # --------------------------------------------------
    # State transitions
    # --------------------------------------------------
    def _get_state_transition_function(self) -> dict:
        return {
            # 0: Need RED_KEY -> State 1
            0: {
                "RED_KEY / (-)": 1,
                # Collectibles are removed from the grid when stepped on.  Do
                # not silently ignore one that is collected too early, or the
                # remaining task can become impossible to complete.
                "BLUE_KEY / (-)": -1,
                "COIN / (-)": -1,
                "GOAL / (-)": -1,
                "/ (-)": 0,
            },

            # 1: Need BLUE_KEY -> State 2
            1: {
                "BLUE_KEY / (-)": 2,
                "COIN / (-)": -1,
                "GOAL / (-)": -1,
                "/ (-)": 1,
            },

            # 2: Coins collected -> State 3
            2: {
                "COIN / (Z)": 3,
                "COIN / (NZ)": 2,
                "/ (-)": 2,
            },

            # 3: Reach GOAL -> Terminal State -1
            3: {
                "GOAL / (-)": -1,
                "/ (-)": 3,
            },
        }

    # --------------------------------------------------
    # Counter transitions
    # --------------------------------------------------
    def _get_counter_transition_function(self) -> dict:
        return {
            0: {
                "RED_KEY / (-)": (0,),
                "BLUE_KEY / (-)": (0,),
                "COIN / (-)": (0,),
                "GOAL / (-)": (0,),
                "/ (-)": (0,),
            },
            1: {
                "BLUE_KEY / (-)": (0,),
                "COIN / (-)": (0,),
                "GOAL / (-)": (0,),
                "/ (-)": (0,),
            },
            # Counter transitions
            2: {
                "COIN / (NZ)": (-1,), 
                "COIN / (Z)": (0,),  
                "/ (-)": (0,),
            },
            3: {
                "GOAL / (-)": (0,),
                "/ (-)": (0,),
            },
        }

    # --------------------------------------------------
    # Reward transitions
    # --------------------------------------------------
    def _get_reward_transition_function(self) -> dict:
        return {
            0: {
                "RED_KEY / (-)": 2.0,
                "BLUE_KEY / (-)": -20.0,
                "COIN / (-)": -20.0,
                "GOAL / (-)": -20.0,
                "/ (-)": -0.01,
            },
            1: {
                "BLUE_KEY / (-)": 2.0,
                "COIN / (-)": -20.0,
                "GOAL / (-)": -20.0,
                "/ (-)": -0.01,
            },

            2: {
                "COIN / (NZ)": 1.0, 
                "COIN / (Z)": 1.0,    
                "/ (-)": -0.01,
            },
            3: {
                "GOAL / (-)": 20.0,
                "/ (-)": -0.01,
            },
        }

    def sample_counter_configurations(self) -> list[tuple[int, ...]]:
        return [(i,) for i in range(6)]

def environment_difficulty(grid, spec):

    valid, metadata = validate_generated_grid(grid, spec)

    hazards = sum(row.count("H") for row in grid)

    total_required = (
        spec["coins"]
        + spec["bells"]
        + sum(spec["keys"].values())
    )

    result = {
        "valid": valid,
        "hazard_count": hazards,
        "hazard_density": hazards / 100,
        "required_object_count": total_required,
    }

    if valid:
        result["shortest_goal_distance"] = metadata["shortest_goal_distance"]
        result["reachable_required_objects"] = metadata["reachable_required_objects"]
    else:
        result["shortest_goal_distance"] = None
        result["reachable_required_objects"] = 0
        result["validation_error"] = metadata

    return result

def crm_complexity(crm):
    states = len(crm._get_state_transition_function())

    transitions = sum(
        len(transitions)
        for transitions in crm._get_state_transition_function().values()
    )

    counter_count = len(crm.c_0)
    counter_configurations = len(crm.sample_counter_configurations())

    return {
        "states": states,
        "transitions": transitions,
        "has_counters": counter_count > 0,
        "counter_count": counter_count,
        "counter_configurations": counter_configurations,
    }

def combined_difficulty(grid, environment_spec, crm=None):
    """
    Combines environment and CRM difficulty into a score in [0,1].
    """
    env_metrics = environment_difficulty(grid, environment_spec)
    
    crm_metrics = None
    if crm is not None:
        crm_metrics = crm_complexity(crm)

    # Normalise path length using env_metrics directly
    if env_metrics["shortest_goal_distance"] is None:
        path_score = 1.0
    else:
        path_score = min(env_metrics["shortest_goal_distance"] / 50, 1.0)

    hazard_score = min(env_metrics["hazard_density"] / 0.20, 1.0)
    object_score = min(env_metrics["required_object_count"] / 15, 1.0)

    crm_score = 0.0
    if crm_metrics is not None:
        state_score = min(crm_metrics["states"] / 10, 1.0)
        counter_count_score = min(crm_metrics["counter_count"] / 3, 1.0)
        counter_presence_score = 1.0 if crm_metrics["has_counters"] else 0.0

        # Counters make the task state larger: an agent must distinguish both
        # CRM state and counter values (for example, how many red keys remain).
        crm_score = (
            0.50 * state_score
            + 0.25 * counter_count_score
            + 0.25 * counter_presence_score
        )

    difficulty = (
        0.40 * path_score
        + 0.20 * hazard_score
        + 0.20 * object_score
        + 0.20 * crm_score
    )

    return {
        "difficulty_score": round(difficulty, 3),
        "environment": env_metrics,
        "crm": crm_metrics,
    }

def _shortest_route_through_group(grid, start_costs, targets):
    """Return the shortest safe route that visits every target in a group."""
    if not targets:
        return start_costs.copy()

    target_count = len(targets)
    distances = {
        position: bfs_distances(grid, position)
        for position in set(start_costs) | set(targets)
    }
    route_costs = {}

    for start, start_cost in start_costs.items():
        # (visited target bitmask, final target index) -> path length
        paths = {}
        for target_index, target in enumerate(targets):
            if target in distances[start]:
                paths[(1 << target_index, target_index)] = (
                    start_cost + distances[start][target]
                )

        for visited_count in range(1, target_count + 1):
            for (mask, last_index), cost in list(paths.items()):
                if mask.bit_count() != visited_count:
                    continue
                for next_index, next_target in enumerate(targets):
                    if mask & (1 << next_index):
                        continue
                    last_target = targets[last_index]
                    if next_target not in distances[last_target]:
                        continue
                    candidate = cost + distances[last_target][next_target]
                    key = (mask | (1 << next_index), next_index)
                    paths[key] = min(paths.get(key, float("inf")), candidate)

        full_mask = (1 << target_count) - 1
        for target_index, target in enumerate(targets):
            cost = paths.get((full_mask, target_index))
            if cost is not None:
                route_costs[target] = min(route_costs.get(target, float("inf")), cost)

    return route_costs


def shortest_task_path_length(grid, spec):
    """Shortest safe path for red keys, blue keys, coins, then the goal."""
    objects = find_grid_objects(grid)
    current_costs = {objects["A"]: 0}

    for targets in (
        objects["keys"]["R"],
        objects["keys"]["B"],
        objects["coins"],
    ):
        current_costs = _shortest_route_through_group(grid, current_costs, targets)

    goal_distances = {
        position: bfs_distances(grid, position).get(objects["G"])
        for position in current_costs
    }
    completed_routes = [
        cost + goal_distances[position]
        for position, cost in current_costs.items()
        if goal_distances[position] is not None
    ]

    if not completed_routes:
        raise ValueError("Task objects and goal must be connected by a safe path.")

    return min(completed_routes)


def task_event_sequence(spec):
    """Events required by this task's CRM, in the task's enforced order."""
    return (
        [Event.RED_KEY] * spec["keys"]["red"]
        + [Event.BLUE_KEY] * spec["keys"]["blue"]
        + [Event.COIN] * spec["coins"]
        + [Event.GOAL]
    )


def calculate_reward_bounds(grid, spec, crm, max_steps):
    """Calculate attainable reward bounds for this CRM and generated grid."""
    state, counters = crm.u_0, crm.c_0
    task_reward = 0.0

    for event in task_event_sequence(spec):
        state, counters, reward_function = crm.transition(state, counters, {event})
        task_reward += reward_function(None, None, None)

    raw_state_transitions = crm._get_state_transition_function()
    raw_reward_transitions = crm._get_reward_transition_function()
    failure_rewards = [
        raw_reward_transitions[state][expression]
        for state, transitions in raw_state_transitions.items()
        for expression, next_state in transitions.items()
        if next_state == -1
    ]

    step_reward = raw_reward_transitions[crm.u_0]["/ (-)"]
    shortest_path = shortest_task_path_length(grid, spec)
    non_event_steps = shortest_path - len(task_event_sequence(spec))
    maximum_reward = task_reward + step_reward * non_event_steps

    # The agent can wait until the final allowed step before triggering the
    # worst task-failure transition.  If the CRM has no failure terminal,
    # repeatedly taking ordinary steps is the worst possible outcome.
    if failure_rewards:
        minimum_reward = min(failure_rewards) + step_reward * (max_steps - 1)
    else:
        minimum_reward = step_reward * max_steps

    return {
        "minimum_reward": minimum_reward,
        "maximum_reward": maximum_reward,
        "shortest_task_path": shortest_path,
    }


def reward_efficiency(episode_reward, minimum_reward, maximum_reward):
    """Normalise an episode return against the task-and-grid reward bounds."""
    reward_range = maximum_reward - minimum_reward
    if reward_range == 0:
        return 1.0 if episode_reward >= maximum_reward else 0.0
    return (episode_reward - minimum_reward) / reward_range


def run_agent(env, q_table, reward_bounds):

    obs, _ = env.reset()

    done = False
    total_reward = 0
    event_sequence = []

    step = 0

    print("=== OPTIMAL POLICY ===")

    while not done:

        env.render()

        state = tuple(obs)

        q = q_table[state]
        if np.all(q == 0):
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q))

        print(
            f"Step {step}: "
            f"State={state}, "
            f"Action={action}"
        )

        obs, reward, terminated, truncated, _ = env.step(action)

        # CrossProduct stores the label(s) produced for the transition.  Only
        # record real events; ordinary movement has an empty proposition set.
        props = env.props
        if props:
            event_sequence.extend(
                sorted(props, key=lambda event: event.name)
            )

        total_reward += reward

        done = terminated or truncated

        step += 1

        print(q_table[state])

    env.render()

    print("Episode reward:", total_reward)
    print("Maximum reward:", reward_bounds["maximum_reward"])
    print(
        "Reward efficiency:",
        reward_efficiency(
            total_reward,
            reward_bounds["minimum_reward"],
            reward_bounds["maximum_reward"],
        ),
    )
    print("Steps:", step)
    print("Event sequence:")
    for index, event in enumerate(event_sequence, start=1):
        print(f"  {index}. {event.name}")

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
            if terminated:
                target = reward
            else:  # truncated or not done
                target = reward + discount_factor * np.max(q_table[next_state])

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

        epsilon = max(
            0.01,
            epsilon * 0.995
        )

    return q_table, episode_returns

spec = {
    "coins": 5,
    "keys": {
        "red": 1,
        "blue": 1,
        "green": 0,
        "yellow": 0
    },
    "bells": 0,
    "hazards": True
}

#Train Agent
ground_env = generate_gridworld(
    seed=42,
    environment_json=json.dumps(spec)
)

lf = GridWorldLabellingFunction(ground_env)

crm = TestCRM()

env = GridWorldCrossProduct(
    ground_env=ground_env,
    crm=crm,
    lf=lf,
    max_steps=200
)

q_table, returns = train_agent(
    env,
    episodes=5000
)

print("Training complete.")
print("Average reward:", np.mean(returns[-100:]))

# Check how many of the states visited during the eval rollout were actually trained
untrained = sum(1 for s in q_table if np.all(q_table[s] == 0))
print(f"{untrained} / {len(q_table)} visited states have all-zero Q-values")

#Run Agent
reward_bounds = calculate_reward_bounds(
    ground_env.grid.tolist(),
    spec,
    crm,
    env.max_steps,
)
run_agent(env, q_table, reward_bounds)

difficulty = combined_difficulty(
    ground_env.grid.tolist(),
    spec,
    crm
)

print(difficulty)

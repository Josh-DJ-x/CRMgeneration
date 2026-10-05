"""Runtime components and parsing helpers for the GridWorld CRM system."""

import ast
import heapq
import json
import math
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from enum import Enum, auto
from functools import lru_cache
from itertools import count, product

import gymnasium as gym
import numpy as np
from google import genai
from gymnasium import spaces
from openai import OpenAI
from pycrm.automaton import CountingRewardMachine
from pycrm.crossproduct import CrossProduct
from pycrm.label import LabellingFunction


# Default state budget shared by task-order validation and shortest-route search.
MAX_SEARCH_STATES = 5_000_000


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class OpenAILLM(LLM):
    """Direct OpenAI Responses API provider for generation or judging."""

    def __init__(self, api_key, model="gpt-4.1-mini"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Set it before running system.py.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.messages = []
        self.resolved_models = []
        self.last_response_model = None

    def generate(self, prompt: str) -> str:
        messages = self.messages + [{"role": "user", "content": prompt}]
        response = self.client.responses.create(
            model=self.model, input=messages, store=False,
            include=["reasoning.encrypted_content"],
        )
        if getattr(response, "status", None) != "completed":
            raise RuntimeError(
                f"OpenAI ({self.model}) response was not completed: "
                f"{getattr(response, 'error', None) or getattr(response, 'incomplete_details', None) or response.status}"
            )
        content = response.output_text
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenAI ({self.model}) returned no answer text.")
        self.messages = messages + list(response.output)
        self.last_response_model = response.model
        self.resolved_models.append(response.model)
        return content

    def reset_history(self):
        self.messages = []


class Gemini(LLM):
    def __init__(self, api_key, model="gemini-3.5-flash"):
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.chat = self.client.chats.create(model=self.model)

    def generate(self, prompt: str) -> str:
        return self.chat.send_message(prompt).text


class Ollama(LLM):
    def __init__(self, model="llama3.2", base_url="http://localhost:11434/v1/"):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.messages = []

    def generate(self, prompt: str) -> str:
        messages = self.messages + [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        content = response.choices[0].message.content
        self.messages = messages + [{"role": "assistant", "content": content}]
        return content


class Groq(LLM):
    def __init__(self, api_key, model="openai/gpt-oss-120b"):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Set it in the same terminal session before running system.py."
            )
        self.model = model
        self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        self.messages = []

    def generate(self, prompt: str) -> str:
        messages = self.messages + [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        content = response.choices[0].message.content
        self.messages = messages + [{"role": "assistant", "content": content}]
        return content


class OpenRouterResponseError(RuntimeError):
    """A provider error or unusable answer returned in an API response."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class OpenRouter(LLM):
    def __init__(self, api_key, model="openrouter/free"):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model
        self.messages = []
        self.resolved_models = []
        self.last_response_model = None

    def generate(self, prompt: str) -> str:
        messages = self.messages + [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        error = getattr(response, "error", None)
        if error:
            code = error.get("code") if isinstance(error, dict) else getattr(error, "code", None)
            detail = error.get("message") if isinstance(error, dict) else getattr(error, "message", None)
            raise OpenRouterResponseError(
                f"OpenRouter judge ({self.model}) returned an error"
                f"{f' [{code}]' if code is not None else ''}: {detail or 'unspecified provider error'}",
                status_code=code,
            )
        choices = getattr(response, "choices", None)
        if not choices:
            raise OpenRouterResponseError(
                f"OpenRouter judge ({self.model}) returned no completion choices. "
                "Provider overload or an invalid response is possible; no cause was supplied."
            )
        choice = choices[0]
        content = getattr(getattr(choice, "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            finish_reason = getattr(choice, "finish_reason", None)
            raise OpenRouterResponseError(
                f"OpenRouter judge ({self.model}) returned no answer text "
                f"(finish_reason={finish_reason!r}). "
                + ("The token budget may have been spent on reasoning; increase the output budget."
                   if finish_reason == "length" else "Try again later or choose another free model.")
            )
        self.last_response_model = getattr(response, "model", None)
        self.resolved_models.append(self.last_response_model or "unknown")
        self.messages = messages + [{"role": "assistant", "content": content}]
        return content

    def reset_history(self):
        self.messages = []


def _parse_trace(trace_text: str, section_name: str) -> tuple[str, ...]:
    """Parse and validate an exact JSON event trace from an LLM response."""
    try:
        trace = json.loads(trace_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{section_name} must be a valid JSON array.") from error

    valid_events = {event.name for event in Event}
    if not isinstance(trace, list) or any(
        not isinstance(event, str) or event not in valid_events for event in trace
    ):
        allowed = ", ".join(sorted(valid_events))
        raise ValueError(f"{section_name} must contain only supported events: {allowed}.")
    return tuple(trace)


def extract_generation_trace(response: str, initial: bool = False):
    """Extract desired trace, optional environment, and CRM from a generation."""
    def decode_section(marker: str, section_name: str):
        match = re.search(re.escape(marker), response, re.IGNORECASE)
        if match is None:
            raise ValueError(f"LLM generation did not contain {section_name}.")
        section_text = response[match.end():].lstrip()
        try:
            value, end = json.JSONDecoder().raw_decode(section_text)
        except json.JSONDecodeError as error:
            raise ValueError(f"{section_name} must contain valid JSON.") from error
        return value, section_text[:end]

    crm_match = re.search(r"<<<CRM>>>", response, re.IGNORECASE)
    if crm_match is None:
        raise ValueError("LLM generation must contain DESIRED_TRACE and CRM sections.")

    trace_match = re.search(r"<<<DESIRED_TRACE>>>\s*(.*?)(?=<<<|\Z)",
                            response, re.DOTALL | re.IGNORECASE)
    if trace_match is None:
        raise ValueError("Initial generation did not contain DESIRED_TRACE.")
    desired_trace = parse_compact_trace(trace_match.group(1))
    crm = response[crm_match.end():].strip()
    environment = None

    environment_match = re.search(r"<<<ENVIRONMENT>>>", response, re.IGNORECASE)
    if environment_match is not None and environment_match.start() < crm_match.start():
        _, environment = decode_section("<<<ENVIRONMENT>>>", "ENVIRONMENT")

    if initial:
        if environment is None:
            raise ValueError("Initial generation did not contain ENVIRONMENT.")
        return desired_trace, crm, environment
    return desired_trace, crm, None


def parse_compact_trace(text: str) -> tuple[str, ...]:
    """Expand EVENT -> EVENT x N into the canonical comparison trace."""
    trace = []
    valid_events = {event.name for event in Event}
    for stage in re.split(r"\s*(?:→|->)\s*", text.strip()):
        stage_match = re.fullmatch(r"([A-Z_]+)(?:\s*(?:×|x|\*)\s*([1-9]\d*))?", stage.strip())
        if stage_match is None:
            raise ValueError(
                "DESIRED_TRACE must use EVENT or EVENT x positive_count, joined by ->."
            )
        event, count_text = stage_match.groups()
        if event not in valid_events:
            raise ValueError(f"DESIRED_TRACE contains unsupported event: {event}.")
        count = int(count_text) if count_text else 1
        # Evaluation has a 200-step budget; this generous bound also prevents
        # malformed model output from allocating an unbounded event array.
        if len(trace) + count > 10000:
            raise ValueError("DESIRED_TRACE exceeds the 10000-event limit.")
        trace.extend([event] * count)
    if not trace or trace[-1] != "GOAL" or trace.count("GOAL") != 1:
        raise ValueError("DESIRED_TRACE must contain one GOAL, at the end.")
    return tuple(trace)


def extract_agent_trace(response: str) -> tuple[str, ...]:
    """Extract the trace that the judge reports from an agent run."""
    match = re.search(r"<<<AGENT_TRACE>>>\s*(\[.*?\])", response, re.DOTALL | re.IGNORECASE)
    if match is None:
        raise ValueError("Judge response did not contain an AGENT_TRACE section.")
    return _parse_trace(match.group(1), "AGENT_TRACE")


def extract_judge_reflection(response: str) -> str:
    """Extract the judge's free-text reflection, if it supplied one."""
    match = re.search(r"<<<REFLECTION>>>\s*(.*)", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match is not None else ""


def extract_refinement_crm(response: str) -> str:
    """Extract the CRM-only response used after initial trace/environment generation."""
    match = re.search(r"<<<CRM>>>\s*(.*)", response, re.DOTALL | re.IGNORECASE)
    if match is None:
        raise ValueError("CRM refinement did not contain a CRM section.")
    return match.group(1).strip()


def traces_match(desired_trace, observed_trace) -> bool:
    """Return whether two traces contain the same events in precisely the same order."""
    return tuple(desired_trace) == tuple(observed_trace)


def crm_accepts_trace(crm, desired_trace):
    """Require the desired trace to terminate only on its final event."""
    return crm_trace_alignment_error(crm, desired_trace) is None


def crm_trace_alignment_error(crm, desired_trace):
    """Explain a failed replay, including states and counter configurations."""
    if crm is None:
        return "CRM did not parse."
    replay = []
    try:
        state, counters = crm.u_0, crm.c_0
        terminal_states = set(getattr(crm, "F", {-1}))
        for index, event_name in enumerate(desired_trace):
            event = event_name if isinstance(event_name, Event) else Event[event_name]
            previous = state
            state, counters, _ = crm.transition(state, counters, {event})
            replay.append(f"{event.name}: State {previous} -> {state}, counters {tuple(map(int, counters))}")
            if state in terminal_states and index != len(desired_trace) - 1:
                return "CRM terminates before final GOAL. Replay: " + "; ".join(replay)
        if state not in terminal_states:
            return ("Final GOAL did not reach terminal. Use GOAL -> -1 directly in the final stage; "
                    "counter conditions refer to positions, and (NZ) tests only the first counter. "
                    "Replay: " + "; ".join(replay))
        return None
    except (LookupError, NameError, TypeError, ValueError, ArithmeticError,
            AttributeError, RuntimeError) as error:
        # pycrm compiles generated expressions; some invalid formulas parse
        # successfully and fail only when executed. Treat these as validation
        # failures so they enter the same correction loop as malformed text.
        detail = f"CRM replay failed ({type(error).__name__}): {error}."
        if isinstance(error, NameError):
            detail += " Use only supported event labels: " + ", ".join(Event.__members__) + "."
        return detail + " Replay: " + "; ".join(replay)


def format_trace(trace) -> str:
    """Return a canonical JSON representation of an event trace."""
    return json.dumps([event.name if isinstance(event, Event) else event for event in trace])


def format_compact_trace(trace):
    """Return a compact run-length representation suitable for error feedback."""
    groups = []
    for event in trace:
        name = event.name if isinstance(event, Event) else event
        if groups and groups[-1][0] == name:
            groups[-1][1] += 1
        else:
            groups.append([name, 1])
    return " -> ".join(f"{name} x {count}" if count > 1 else name for name, count in groups)


def extraction(response: str, initial: bool = False):
    """Backward-compatible alias for trace-based generation extraction."""
    return extract_generation_trace(response, initial)


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

        reward = -0.1

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
            if self.grid[row][col] == "S":
                self.last_event = Event.BELL
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
        elif symbol == "H":
            self.last_event = Event.HAZARD

    def is_terminal(self):
        row, col = self.agent_position
        symbol = self.grid[row][col]

        # Goal completion is handled by CRM terminal state (-1).
        if symbol == "H":
            return False

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

def _safe_distances(grid, start, blocked_positions):
    """Find safe distances while treating selected object cells as blocked."""
    height, width = len(grid), len(grid[0])
    queue = deque([start])
    distances = {start: 0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        row, col = queue.popleft()
        for row_delta, col_delta in directions:
            next_position = (row + row_delta, col + col_delta)
            next_row, next_col = next_position
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if next_position in distances or next_position in blocked_positions:
                continue
            if grid[next_row][next_col] in {"#", "H"}:
                continue
            distances[next_position] = distances[(row, col)] + 1
            queue.append(next_position)
    return distances


class EnvironmentGenerationError(ValueError):
    """A bounded environment-generation or order search could not finish."""


def validate_task_order(grid, task_trace, max_search_states=None, deadline=None):
    """Check that the exact task trace can be completed without early pickups."""
    if max_search_states is None:
        max_search_states = MAX_SEARCH_STATES
    objects = find_grid_objects(grid)
    event_targets = {
        Event.COIN: objects["coins"],
        Event.RED_KEY: objects["keys"]["R"],
        Event.BLUE_KEY: objects["keys"]["B"],
        Event.GREEN_KEY: objects["keys"]["E"],
        Event.YELLOW_KEY: objects["keys"]["Y"],
        Event.BELL: objects["bells"],
        Event.GOAL: [objects["G"]],
    }
    collectible_positions = set(objects["coins"])
    collectible_positions.update(
        position for positions in objects["keys"].values() for position in positions
    )
    trace = [event if isinstance(event, Event) else Event[event] for event in task_trace]
    if deadline is None:
        deadline = time.monotonic() + 10.0
    states = [(0, objects["A"], frozenset())]
    visited = set()
    collectible_events = {
        Event.COIN, Event.RED_KEY, Event.BLUE_KEY, Event.GREEN_KEY, Event.YELLOW_KEY,
    }

    # Depth-first search can return the first valid order instead of building
    # every combination of collected objects at each event in a long trace.
    while states:
        if time.monotonic() >= deadline:
            raise EnvironmentGenerationError("Task-order validation reached its time limit.")
        trace_index, position, collected = states.pop()
        state = (trace_index, position, collected)
        if state in visited:
            continue
        if trace_index == len(trace):
            return True
        if len(visited) >= max_search_states:
            raise EnvironmentGenerationError(
                f"Task-order validation reached its search limit ({max_search_states} states)."
            )
        visited.add(state)
        event = trace[trace_index]
        successors = []
        for target in event_targets.get(event, []):
            if time.monotonic() >= deadline:
                raise EnvironmentGenerationError("Task-order validation reached its time limit.")
            if target is None or (event in collectible_events and target in collected):
                continue
            # Uncollected keys and coins must not emit an early event.
            blocked = collectible_positions - set(collected)
            blocked.discard(target)
            if target != objects["G"]:
                blocked.add(objects["G"])
            distances = _safe_distances(grid, position, blocked)
            if target not in distances:
                continue
            next_collected = collected | ({target} if event in collectible_events else set())
            successors.append((distances[target], target, frozenset(next_collected)))
        # Visit nearby objects first to find a feasible order sooner.
        successors.sort(key=lambda candidate: candidate[0], reverse=True)
        states.extend((trace_index + 1, target, collected)
                      for _, target, collected in successors)
    return False


def validate_generated_grid(grid, spec, expected_wall_count=None, task_trace=None,
                            task_order_deadline=None):
    """
    Generic validation for a generated environment.

    Checks:
      - Object counts match the specification.
      - Every required object is reachable from A.
      - Goal is reachable from A.

    Returns (valid, metadata).
    """

    objects = find_grid_objects(grid)

    wall_count = sum(row.count("#") for row in grid)
    if expected_wall_count is not None and wall_count != expected_wall_count:
        return False, "Incorrect number of walls."

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

    if task_trace is not None and not validate_task_order(
        grid, task_trace, deadline=task_order_deadline
    ):
        return False, "Task trace cannot be safely completed in order."

    metadata = {
        "reachable_required_objects": reachable_required_objects,
        "wall_count": wall_count,
    }

    return True, metadata

def generate_gridworld(seed, environment_json, task_trace=None, max_attempts=1000,
                       time_limit_seconds=60.0):

    spec = parse_environment(environment_json)
    rng = np.random.default_rng(seed)
    if max_attempts < 1 or time_limit_seconds <= 0:
        raise ValueError("Environment generation limits must be positive.")
    generation_started = time.monotonic()
    deadline = generation_started + time_limit_seconds
    warned_slow_generation = False

    def generation_time():
        nonlocal warned_slow_generation
        now = time.monotonic()
        if not warned_slow_generation and now - generation_started >= 30.0:
            print(
                "Warning: Grid generation has taken 30 seconds; "
                f"the time limit is {time_limit_seconds:g} seconds.",
                flush=True,
            )
            warned_slow_generation = True
        return now

    last_reason = "No valid layout found."

    SIZE = 10

    for attempt in range(1, max_attempts + 1):
        if generation_time() >= deadline:
            raise EnvironmentGenerationError(
                f"Environment generation reached its {time_limit_seconds:g}-second time limit. "
                f"Last validation result: {last_reason}"
            )

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
                if generation_time() >= deadline:
                    raise EnvironmentGenerationError(
                        f"Environment generation reached its {time_limit_seconds:g}-second "
                        "time limit while placing objects."
                    )

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
        # Walls are always present, but are not part of the task spec.
        # Place them last so they cannot overwrite task objects, then validate
        # that they have not disconnected any required object or the goal.
        # ----------------------------------------------------
        wall_count = rng.integers(10, 15)  # 10–14 walls per 10x10 grid.
        place_objects("#", wall_count)

        try:
            valid, metadata = validate_generated_grid(
                grid, spec, wall_count, task_trace,
                task_order_deadline=min(deadline, generation_time() + 2.0),
            )
        except EnvironmentGenerationError as error:
            generation_time()
            last_reason = str(error)
            continue

        generation_time()
        if valid:
            return GridWorld(grid)
        last_reason = str(metadata)

    raise EnvironmentGenerationError(
        f"Environment generation could not find a valid layout after {max_attempts} attempts. "
        f"Last validation result: {last_reason}"
    )

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
    # Reserve space for the maximum hazards and default walls. Walls are an
    # environment feature, not a user-configurable specification field.
    capacity = SIZE * SIZE - 2 - 14

    total_objects = (
        environment["coins"]
        + environment["bells"]
        + sum(environment["keys"].values())
        + (7 if environment["hazards"] else 0)
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
        self._collectible_positions = tuple(
            (row, col)
            for row, cells in enumerate(ground_env.grid)
            for col, symbol in enumerate(cells)
            if symbol in {"C", "R", "B", "E", "Y"}
        )

    def _get_obs(self, ground_obs, u, c):
        """Include consumed objects so different collection histories stay distinct."""
        machine_obs = super()._get_obs(ground_obs, u, c)
        collected = np.array([
            position in self.ground_env.collected_positions
            for position in self._collectible_positions
        ], dtype=np.float32)
        return np.concatenate((machine_obs, collected))

    def to_ground_obs(self, obs):
        return np.asarray(obs)[:2]

    def get_observation(self):
        return self._get_obs(self.ground_env.get_observation(), self.u, self.c)

    def progress_potential(self):
        """Negative distance to an available event rewarded by the current CRM."""
        if self.u in self.crm.F:
            return 0.0
        _, _, default_reward = self.crm.transition(self.u, self.c, set())
        baseline = _reward_value(default_reward)
        targets, blocked = set(), set()
        for row, cells in enumerate(self.ground_env.grid):
            for col, symbol in enumerate(cells):
                position = (row, col)
                if symbol == "#":
                    blocked.add(position)
                    continue
                event = GRID_OBJECTS.get(symbol)
                if event is None or position in self.ground_env.collected_positions:
                    continue
                _, _, reward = self.crm.transition(self.u, self.c, {event})
                if _reward_value(reward) > baseline:
                    targets.add(position)
                else:
                    blocked.add(position)
        start = self.ground_env.agent_position
        queue, visited = deque([(start, 0)]), {start}
        while queue:
            position, distance = queue.popleft()
            if position in targets:
                # Bells require an interaction after reaching their cell.
                return -float(distance + (self.ground_env.grid[position] == "S"))
            row, col = position
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbour = (row + dr, col + dc)
                if (0 <= neighbour[0] < self.ground_env.height
                        and 0 <= neighbour[1] < self.ground_env.width
                        and neighbour not in visited and neighbour not in blocked):
                    visited.add(neighbour)
                    queue.append((neighbour, distance + 1))
        return -float(self.max_steps)

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

def _saturating_score(value, half_score):
    """Map a non-negative quantity to [0, 1) with no arbitrary hard cap."""
    value = max(float(value), 0.0)
    return value / (value + half_score)


def environment_difficulty(grid, spec, desired_trace=None):

    valid, metadata = validate_generated_grid(grid, spec)

    hazards = sum(row.count("H") for row in grid)
    walls = sum(row.count("#") for row in grid)

    total_required = (
        spec["coins"]
        + spec["bells"]
        + sum(spec["keys"].values())
    )

    result = {
        "valid": valid,
        "hazard_count": hazards,
        "hazard_density": hazards / 100,
        "wall_count": walls,
        "wall_density": walls / 100,
        "required_object_count": total_required,
    }

    if valid:
        result["reachable_required_objects"] = metadata["reachable_required_objects"]
    else:
        result["reachable_required_objects"] = 0
        result["validation_error"] = metadata

    route = shortest_valid_trace_events(grid, desired_trace) if valid and desired_trace else None
    result["minimum_steps_to_complete_trace"] = len(route) if route is not None else None

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


def crm_specification(crm):
    """Return a JSON-serialisable description of an executable CRM."""
    return {
        "u0": crm.u_0,
        "c0": list(crm.c_0),
        "state_transitions": {
            str(state): dict(transitions)
            for state, transitions in crm._get_state_transition_function().items()
        },
        "counter_transitions": {
            str(state): {
                expression: list(values)
                for expression, values in transitions.items()
            }
            for state, transitions in crm._get_counter_transition_function().items()
        },
        "reward_transitions": {
            str(state): {
                expression: _reward_value(reward)
                for expression, reward in transitions.items()
            }
            for state, transitions in crm._get_reward_transition_function().items()
        },
    }

def combined_difficulty(grid, environment_spec, crm=None, desired_trace=None):
    """Task workload calibration; CRM complexity is diagnostic only.

    Three ordered keys with a 38-step route score 0.180. The 48-event
    reference task scores about 0.90, depending on layout and hazards.
    This is a calibrated task descriptor, not an empirical failure probability.
    """
    env_metrics = environment_difficulty(grid, environment_spec, desired_trace)
    crm_metrics = crm_complexity(crm) if crm is not None else None
    events = [event.name if isinstance(event, Event) else event
              for event in (desired_trace or ())]
    events = [event for event in events if event != "GOAL"]
    event_count = len(events) if desired_trace is not None else env_metrics["required_object_count"]
    stages = sum(index == 0 or event != events[index - 1]
                 for index, event in enumerate(events))
    minimum_steps = env_metrics["minimum_steps_to_complete_trace"]
    components = {
        "required_events": float(event_count),
        "ordered_stage_changes": 0.5 * max(stages - 1, 0),
        "route_steps": (minimum_steps or 0) / 50.0,
        "hazards": 4.0 * _saturating_score(env_metrics["hazard_count"], 5),
    }
    workload = sum(components.values())
    difficulty = -math.expm1(-workload / 24.0)
    return {
        "difficulty_score": round(difficulty, 3),
        "formula": "1 - exp(-workload / 24)",
        "workload": workload,
        "workload_components": components,
        "required_event_count": event_count,
        "ordered_stage_count": stages,
        "route_length_known": minimum_steps is not None,
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
        task_reward += _reward_value(reward_function)

    raw_state_transitions = crm._get_state_transition_function()
    raw_reward_transitions = crm._get_reward_transition_function()
    failure_rewards = [
        _reward_value(raw_reward_transitions[state][expression])
        for state, transitions in raw_state_transitions.items()
        for expression, next_state in transitions.items()
        if next_state == -1
    ]

    step_reward = _reward_value(raw_reward_transitions[crm.u_0]["/ (-)"])
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


def step_efficiency(minimum_steps, steps_taken):
    """Measure route efficiency relative to the exact shortest task route."""
    return minimum_steps / steps_taken if steps_taken > 0 else 0.0


def _reward_value(reward):
    """Return a numeric reward from either a CRM callback or a raw value."""
    return float(reward(None, None, None) if callable(reward) else reward)

class RouteSearchLimitError(ValueError):
    """An exact route search exceeded its budget without proving infeasibility."""


def shortest_valid_trace_events(grid, desired_trace, max_search_states=None,
                               time_limit_seconds=30.0):
    """Find an exact shortest route; reuse it for difficulty and reward bounds."""
    if max_search_states is None:
        max_search_states = MAX_SEARCH_STATES
    return _shortest_valid_trace_events(
        tuple(tuple(row) for row in grid), tuple(desired_trace),
        max_search_states, time_limit_seconds,
    )


@lru_cache(maxsize=8)
def _shortest_valid_trace_events(grid, desired_trace, max_search_states,
                                time_limit_seconds):
    """Return the event/no-event sequence of the shortest legal task route.

    The search models collection, bell interaction, walls, hazards, and the exact
    required trace.  It therefore cannot collect a later object or enter the goal
    early. ``None`` means the search proved there is no route; budget exhaustion
    raises ``RouteSearchLimitError``. Movement between events is collapsed into
    shortest safe paths, and consecutive interactions use the same bell.
    """
    objects = find_grid_objects(grid)
    start = objects["A"]
    if start is None:
        return None
    trace = tuple(event if isinstance(event, Event) else Event[event] for event in desired_trace)
    collectible_events = {
        "C": Event.COIN, "R": Event.RED_KEY, "B": Event.BLUE_KEY,
        "E": Event.GREEN_KEY, "Y": Event.YELLOW_KEY,
    }
    collectible_positions = [
        (row, col)
        for row, row_values in enumerate(grid)
        for col, symbol in enumerate(row_values)
        if symbol in collectible_events
    ]
    collectible_index = {position: index for index, position in enumerate(collectible_positions)}
    height, width = len(grid), len(grid[0])
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    remaining_bells = [0] * (len(trace) + 1)
    for index in range(len(trace) - 1, -1, -1):
        remaining_bells[index] = remaining_bells[index + 1] + (trace[index] == Event.BELL)

    def heuristic(position, trace_index):
        # Every remaining pickup/goal requires a movement step; each bell event
        # needs an interaction. Distance to the goal is another movement bound.
        movement = len(trace) - trace_index - remaining_bells[trace_index]
        if objects["G"] is not None and Event.GOAL in trace[trace_index:]:
            goal_row, goal_col = objects["G"]
            movement = max(movement, abs(position[0] - goal_row) + abs(position[1] - goal_col))
        return movement + remaining_bells[trace_index]

    deadline = time.monotonic() + time_limit_seconds
    initial = (start, 0, 0)  # position, trace index, collected-object bitmask
    serial = count()
    queue = [(heuristic(start, 0), 0, next(serial), initial)]
    costs = {initial: 0}
    parents = {initial: None}

    while queue:
        if time.monotonic() >= deadline:
            raise RouteSearchLimitError(
                f"Shortest-route search reached its {time_limit_seconds:g}-second time limit; "
                "a valid environment may still exist."
            )
        _, cost, _, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        position, trace_index, collected_mask = current
        if trace_index == len(trace):
            segments = []
            while parents[current] is not None:
                current, segment = parents[current]
                segments.append(segment)
            return tuple(event for segment in reversed(segments) for event in segment)

        # Reach every possible next event through cells which emit no event.
        # An uncollected object or the goal can only be the endpoint of a path.
        distances = {position: 0}
        movement_queue = deque([position])
        endpoints = {}
        event = trace[trace_index]
        while movement_queue:
            row, col = movement_queue.popleft()
            for dr, dc in directions:
                target = (row + dr, col + dc)
                nr, nc = target
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                symbol = grid[nr][nc]
                if symbol in {"#", "H"}:
                    continue
                bit = 1 << collectible_index[target] if target in collectible_index else 0
                if (bit and not collected_mask & bit) or symbol == "G":
                    emitted = collectible_events.get(symbol, Event.GOAL)
                    if emitted == event:
                        endpoints.setdefault(target, distances[(row, col)] + 1)
                    continue
                if target not in distances:
                    distances[target] = distances[(row, col)] + 1
                    movement_queue.append(target)

        repetitions = 1
        if event == Event.BELL:
            while trace_index + repetitions < len(trace) and trace[trace_index + repetitions] == event:
                repetitions += 1
            endpoints = {target: distances[target] for target in objects["bells"] if target in distances}
        for target, distance in endpoints.items():
            next_mask = collected_mask
            if event in collectible_events.values():
                next_mask |= 1 << collectible_index[target]
            successor = (target, trace_index + repetitions, next_mask)
            if event == Event.BELL:
                segment = (None,) * distance + (event.name,) * repetitions
            else:
                segment = (None,) * (distance - 1) + (event.name,)
            next_cost = cost + len(segment)
            if next_cost >= costs.get(successor, math.inf):
                continue
            if successor not in costs and len(costs) >= max_search_states:
                raise RouteSearchLimitError(
                    f"Shortest-route search reached its search limit ({max_search_states} states); "
                    "a valid environment may still exist."
                )
            costs[successor] = next_cost
            parents[successor] = (current, segment)
            heapq.heappush(queue, (next_cost + heuristic(target, trace_index + repetitions),
                                   next_cost, next(serial), successor))
    return None


def calculate_trace_reward_bounds(crm, desired_trace, max_steps, grid):
    """Calculate bounds using the shortest physically attainable ordered route."""
    route_events = shortest_valid_trace_events(grid, desired_trace)
    if route_events is None:
        raise ValueError("No ordered route exists for the desired trace.")

    state, counters = crm.u_0, crm.c_0
    maximum_reward = 0.0
    for event_name in route_events:
        props = {Event[event_name]} if event_name else set()
        state, counters, reward_function = crm.transition(state, counters, props)
        maximum_reward += _reward_value(reward_function)

    reward_transitions = crm._get_reward_transition_function()
    all_rewards = [
        _reward_value(reward)
        for transitions in reward_transitions.values()
        for reward in transitions.values()
    ]
    default_reward = _reward_value(reward_transitions[crm.u_0].get("/ (-)", 0.0))
    failure_rewards = [
        _reward_value(reward_transitions[state][expression])
        for state, transitions in crm._get_state_transition_function().items()
        for expression, next_state in transitions.items()
        if next_state == -1 and expression != "GOAL / (-)"
    ]
    return {
        "minimum_reward": min(failure_rewards, default=min(all_rewards)) + default_reward * (max_steps - 1),
        "maximum_reward": maximum_reward,
        "shortest_valid_steps": len(route_events),
    }


def _grid_frame(env):
    """Capture a printable ground-grid frame without rendering state diagnostics."""
    ground_env = getattr(env, "ground_env", env)
    display = ground_env.grid.copy()
    if ground_env.start_position is not None:
        row, col = ground_env.start_position
        display[row][col] = "."
    for row, col in ground_env.collected_positions:
        if display[row][col] != "G":
            display[row][col] = "."
    row, col = ground_env.agent_position
    display[row][col] = "A"
    return "\n".join(" ".join(row) for row in display)


def print_grid_progression(frames):
    """Print previously captured grid frames, with no state or action details."""
    for frame in frames:
        print(frame)
        print()


def valid_gridworld_actions(env):
    """Return actions that can change the GridWorld state without hitting a wall.

    This uses map information already available to the agent/environment. Hazards
    remain selectable so tasks that intentionally involve hazards retain their
    semantics; only impossible wall/boundary moves and useless interaction actions
    are excluded.
    """
    ground_env = getattr(env, "ground_env", env)
    row, col = ground_env.agent_position
    grid = ground_env.grid
    height, width = grid.shape
    actions = []
    for action, (row_delta, col_delta) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        next_row, next_col = row + row_delta, col + col_delta
        if 0 <= next_row < height and 0 <= next_col < width and grid[next_row][next_col] != "#":
            actions.append(action)
    if grid[row][col] == "S":
        actions.append(4)
    return actions or list(range(ground_env.action_space.n))


def _best_valid_action(q_values, actions, rng):
    """Break Q-value ties randomly to avoid a persistent directional bias."""
    values = np.asarray(q_values)[actions]
    candidates = np.asarray(actions)[np.flatnonzero(values == values.max())]
    return int(rng.choice(candidates))


def run_agent(env, q_table, reward_bounds, capture_progression=False):
    """Run one greedy evaluation episode and return raw, judge-ready run information."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    event_sequence = []
    frames = []
    bell_rings = []
    steps = 0

    while not done:
        if capture_progression:
            frames.append(_grid_frame(env))
        state = tuple(obs)
        q = q_table[state]
        action = _best_valid_action(q, valid_gridworld_actions(env), np.random)
        obs, reward, terminated, truncated, _ = env.step(action)
        props = env.props
        if props:
            event_sequence.extend(sorted(props, key=lambda event: event.name))
            if Event.BELL in props:
                ground_env = getattr(env, "ground_env", env)
                bell_rings.append({
                    "frame_index": len(frames),
                    "ring_number": len(bell_rings) + 1,
                    "position": tuple(int(value) for value in ground_env.agent_position),
                })
        total_reward += reward
        done = terminated or truncated
        steps += 1

    if capture_progression:
        frames.append(_grid_frame(env))
    return {
        "trace": tuple(event.name for event in event_sequence),
        "episode_reward": total_reward,
        "minimum_reward": reward_bounds["minimum_reward"],
        "maximum_reward": reward_bounds["maximum_reward"],
        "reward_efficiency": reward_efficiency(
            total_reward, reward_bounds["minimum_reward"], reward_bounds["maximum_reward"]
        ),
        "steps": steps,
        "minimum_steps": reward_bounds["shortest_valid_steps"],
        "step_efficiency": step_efficiency(reward_bounds["shortest_valid_steps"], steps),
        "terminated": terminated,
        "truncated": truncated,
        "grid_progression": frames,
        "bell_rings": bell_rings,
    }

def train_agent(
    env,
    episodes=5000,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    min_epsilon=0.05,
    epsilon_decay=0.999,
    seed=42,
    reward_shaping=True,
    evaluation_interval=1000,
    stop_training=None,
):
    """Train continuously, optionally checking a stop callback between episodes.

    stop_training(q_table, completed_episodes) may evaluate the environment
    and returns True when training should stop. The next episode resets it.
    """
    if stop_training is not None and evaluation_interval <= 0:
        raise ValueError("evaluation_interval must be positive")
    q_table = defaultdict(
        lambda: np.zeros(env.action_space.n)
    )

    episode_returns = []
    rng = np.random.default_rng(seed)

    for episode in range(episodes):

        obs, _ = env.reset()

        state = tuple(obs)
        potential = env.progress_potential() if reward_shaping and isinstance(env, GridWorldCrossProduct) else 0.0

        episode_return = 0.0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            # Epsilon-greedy action selection
            valid_actions = valid_gridworld_actions(env)
            if rng.random() < epsilon:
                action = int(rng.choice(valid_actions))

            else:
                action = _best_valid_action(q_table[state], valid_actions, rng)

            next_obs, reward, terminated, truncated, _ = env.step(
                action
            )

            next_state = tuple(next_obs)

            episode_return += reward
            next_potential = (
                env.progress_potential()
                if reward_shaping and isinstance(env, GridWorldCrossProduct) and not (terminated or truncated)
                else 0.0
            )
            learning_reward = reward + discount_factor * next_potential - potential

            # Q-learning update
            if terminated or truncated:
                target = learning_reward
            else:  # truncated or not done
                target = learning_reward + discount_factor * np.max(
                    q_table[next_state][valid_gridworld_actions(env)]
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
            potential = next_potential

        episode_returns.append(episode_return)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        completed_episodes = episode + 1
        if (stop_training is not None
                and (completed_episodes % evaluation_interval == 0
                     or completed_episodes == episodes)
                and stop_training(q_table, completed_episodes)):
            break

    return q_table, episode_returns

class GeneratedCRM(CountingRewardMachine):
    """A Counting Reward Machine instantiated from a validated LLM specification."""

    def __init__(self, specification):
        self._specification = specification
        super().__init__(env_prop_enum=Event)

    @property
    def u_0(self):
        return self._specification["u0"]

    @property
    def c_0(self):
        return self._specification["c0"]

    @property
    def encoded_configuration_size(self):
        # Includes the terminal configuration and provides a stable observation size.
        return max(2, len(self._specification["state_transitions"]) + 1)

    def _get_state_transition_function(self):
        return self._specification["state_transitions"]

    def _get_counter_transition_function(self):
        return self._specification["counter_transitions"]

    def _get_reward_transition_function(self):
        return self._specification["reward_transitions"]

    def sample_counter_configurations(self):
        # A bounded sample is sufficient for encoding and supports ordinary count tasks.
        counter_count = len(self.c_0)
        return list(product(range(11), repeat=counter_count)) if counter_count else [()]


def crm_parse(crm_text: str):
    """Parse the documented CRM text format into an executable reward machine."""
    try:
        u0_match = re.search(r"^\s*u0\s*=\s*(-?\d+)\s*$", crm_text, re.MULTILINE | re.IGNORECASE)
        c0_match = re.search(r"^\s*c0\s*=\s*(\([^\n]*\))\s*$", crm_text, re.MULTILINE | re.IGNORECASE)
        if u0_match is None or c0_match is None:
            raise ValueError("CRM must define both `u0 = <state>` and `c0 = (<counters>)`.")
        counters = ast.literal_eval(c0_match.group(1))
        if not isinstance(counters, tuple) or not all(isinstance(value, int) for value in counters):
            raise ValueError("CRM c0 must be a tuple of integers.")

        specification = {
            "u0": int(u0_match.group(1)),
            "c0": counters,
            "state_transitions": {},
            "counter_transitions": {},
            "reward_transitions": {},
        }
        section_lookup = {
            "state transitions": "state_transitions",
            "counter transitions": "counter_transitions",
            "reward transitions": "reward_transitions",
        }
        current_section = None
        current_state = None

        for raw_line in crm_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalised = re.sub(r"\s+", " ", line).lower()
            if normalised in section_lookup:
                current_section = section_lookup[normalised]
                current_state = None
                continue
            state_match = re.fullmatch(r"State\s+(-?\d+)", line, re.IGNORECASE)
            if state_match:
                if current_section is None:
                    raise ValueError("A CRM state was declared before a transition section.")
                current_state = int(state_match.group(1))
                specification[current_section][current_state] = {}
                continue
            if "->" not in line:
                continue
            if current_section is None or current_state is None:
                raise ValueError("Every CRM transition must occur inside a section and state.")
            expression, value = (part.strip() for part in line.split("->", 1))
            if not expression:
                expression = "/ (-)"
            if expression.upper() == "DEFAULT / (-)":
                expression = "/ (-)"
            if current_section == "state_transitions":
                terminal = value.lower() in {"terminal", "terminal state", "-1"}
                state_value = -1 if terminal else int(re.sub(r"^State\s+", "", value, flags=re.IGNORECASE))
                specification[current_section][current_state][expression] = state_value
            elif current_section == "counter_transitions":
                try:
                    counter_value = ast.literal_eval(value)
                except (SyntaxError, ValueError) as error:
                    raise ValueError("Counter updates must be integer delta tuples, e.g. (-1,0); "
                                     "Python expressions and assignments are unsupported.") from error
                if not isinstance(counter_value, tuple) or not all(isinstance(item, int) for item in counter_value):
                    raise ValueError("Counter transition values must be integer tuples.")
                specification[current_section][current_state][expression] = counter_value
            else:
                specification[current_section][current_state][expression] = float(value)

        states = set(specification["state_transitions"])
        if specification["u0"] not in states or not states:
            raise ValueError("CRM initial state must be declared in State Transitions.")
        for section in ("counter_transitions", "reward_transitions"):
            if set(specification[section]) != states:
                raise ValueError(f"CRM {section.replace('_', ' ')} must define every state.")
        for state in states:
            expressions = set(specification["state_transitions"][state])
            if expressions != set(specification["counter_transitions"][state]) or expressions != set(specification["reward_transitions"][state]):
                raise ValueError(f"CRM transition expressions must match across all sections for State {state}.")
            for expression in expressions:
                delta = specification["counter_transitions"][state][expression]
                if len(delta) != len(counters):
                    raise ValueError(f"State {state}: counter update {delta} must have {len(counters)} positions, matching c0.")
                condition = expression.rsplit("/", 1)[-1].strip().strip("()")
                if condition != "-" and len(condition.split(",")) > len(counters):
                    raise ValueError(f"State {state}: counter condition ({condition}) can have at most {len(counters)} positions, matching c0; (-) ignores all counters.")
        return GeneratedCRM(specification)
    except (SyntaxError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid CRM format: {error}") from error

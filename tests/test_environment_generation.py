"""Offline checks for bounded grid generation and ordered pickup searches."""

import json
import random
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system_functions import (
    EnvironmentGenerationError, RouteSearchLimitError, calculate_trace_reward_bounds,
    generate_gridworld, shortest_valid_trace_events, validate_task_order,
    validate_generated_grid,
)


class EnvironmentGenerationTests(unittest.TestCase):
    def setUp(self):
        self.spec = {"coins": 0, "bells": 0, "hazards": False,
                     "keys": {"red": 0, "blue": 0, "green": 0, "yellow": 0}}

    def test_long_coin_and_bell_trace_finds_first_valid_order(self):
        grid = [list("A" + "C" * 12 + "S" + "C" * 15 + "E" * 4 + "R" * 2 + "G")]
        trace = (["COIN"] * 12 + ["BELL"] * 37 + ["COIN"] * 15
                 + ["GREEN_KEY"] * 4 + ["RED_KEY"] * 2 + ["GOAL"])
        self.assertTrue(validate_task_order(grid, trace))

    def test_search_backtracks_when_nearest_coin_blocks_later_key(self):
        grid = [list(row) for row in ("AC##", "C#EG", "...#")]
        self.assertTrue(validate_task_order(grid, ["COIN", "GREEN_KEY", "COIN", "GOAL"]))

    def test_search_rejects_unavoidable_early_pickup(self):
        self.assertFalse(validate_task_order([list("ACEG")], ["GREEN_KEY", "COIN", "GOAL"]))

    def test_generation_rejects_solvable_collectible_dependency(self):
        # Either coin can be first, but only the entrance coin allows the key
        # next. A solvable trace alone must not qualify this layout.
        grid = [list(row) for row in ("ACB#", ".###", "C..G")]
        spec = dict(self.spec, coins=2,
                    keys=dict(self.spec["keys"], blue=1))
        trace = ["COIN", "BLUE_KEY", "COIN", "GOAL"]
        self.assertTrue(validate_task_order(grid, trace))
        valid, reason = validate_generated_grid(
            grid, spec, task_trace=trace, independent_access=True)
        self.assertFalse(valid)
        self.assertIn("requires crossing", reason)

    def test_generation_accepts_independently_accessible_collectibles(self):
        grid = [list(row) for row in ("ACB.", "....", "C..G")]
        spec = dict(self.spec, coins=2,
                    keys=dict(self.spec["keys"], blue=1))
        self.assertTrue(validate_generated_grid(
            grid, spec, task_trace=["COIN", "BLUE_KEY", "COIN", "GOAL"],
            independent_access=True,
        )[0])

    def test_generation_rejects_bell_or_goal_behind_collectible(self):
        for row, bells in (("ACSG", 1), ("ACG", 0)):
            with self.subTest(row=row):
                valid, reason = validate_generated_grid(
                    [list(row)], dict(self.spec, coins=1, bells=bells),
                    independent_access=True)
                self.assertFalse(valid)
                self.assertIn("requires crossing", reason)

    def test_order_search_has_state_and_time_limits(self):
        grid = [list("ACG")]
        with self.assertRaisesRegex(EnvironmentGenerationError, "search limit"):
            validate_task_order(grid, ["COIN", "GOAL"], max_search_states=1)
        with self.assertRaisesRegex(EnvironmentGenerationError, "time limit"):
            validate_task_order(grid, ["COIN", "GOAL"], deadline=0)

    def test_rejected_layouts_stop_after_attempt_limit(self):
        with patch("system_functions.validate_generated_grid", return_value=(False, "Goal unreachable.")) as validate:
            with self.assertRaisesRegex(EnvironmentGenerationError, "after 3 attempts.*Goal unreachable"):
                generate_gridworld(42, json.dumps(self.spec), max_attempts=3)
        self.assertEqual(validate.call_count, 3)

    def test_order_search_limit_allows_retry_with_another_layout(self):
        with patch("system_functions.validate_generated_grid", side_effect=[
            EnvironmentGenerationError("search limit"), (True, {}),
        ]) as validate:
            env = generate_gridworld(42, json.dumps(self.spec), max_attempts=2)
        self.assertEqual(env.grid.shape, (10, 10))
        self.assertEqual(validate.call_count, 2)

    def test_generation_deadline_stops_before_another_attempt(self):
        with patch("system_functions.time.monotonic", side_effect=[0, 61]):
            with self.assertRaisesRegex(EnvironmentGenerationError, "60-second time limit"):
                generate_gridworld(42, json.dumps(self.spec))

    def test_generation_warns_once_at_30_seconds_and_keeps_trying(self):
        clock = [0.0]
        attempts = [0]
        def validate(*args, **kwargs):
            attempts[0] += 1
            clock[0] = 30.0 if attempts[0] == 1 else 45.0
            return (attempts[0] == 2, {})
        with (patch("system_functions.time.monotonic", side_effect=lambda: clock[0]),
              patch("system_functions.validate_generated_grid", side_effect=validate),
              patch("builtins.print") as output):
            env = generate_gridworld(42, json.dumps(self.spec))
        self.assertEqual(env.grid.shape, (10, 10))
        self.assertEqual(attempts[0], 2)
        output.assert_called_once_with(
            "Warning: Grid generation has taken 30 seconds; the time limit is 60 seconds.",
            flush=True,
        )

    def test_shortest_route_search_limits_are_not_reported_as_no_route(self):
        with self.assertRaisesRegex(RouteSearchLimitError, "search limit.*may still exist"):
            shortest_valid_trace_events([list("ACG")], ["COIN", "GOAL"], max_search_states=1)
        with self.assertRaisesRegex(RouteSearchLimitError, "time limit.*may still exist"):
            shortest_valid_trace_events([list("ACG")], ["COIN", "GOAL"], time_limit_seconds=0)
        self.assertIsNone(shortest_valid_trace_events([list("A#G")], ["GOAL"]))

    def test_validation_cannot_cross_goal_before_finishing_pickups(self):
        self.assertFalse(validate_task_order([list("AGC")], ["COIN", "GOAL"]))
        self.assertIsNone(shortest_valid_trace_events([list("AGC")], ["COIN", "GOAL"]))

    def test_shortest_route_matches_independent_cell_search_on_small_grids(self):
        # Compare event-to-event search with a simple movement-by-movement BFS.
        def reference_length(grid, trace):
            queue = deque([((0, 0), 0, frozenset(), 0)])
            visited = set()
            while queue:
                position, index, collected, steps = queue.popleft()
                state = (position, index, collected)
                if state in visited:
                    continue
                visited.add(state)
                if index == len(trace):
                    return steps
                row, col = position
                if grid[row][col] == "S" and trace[index] == "BELL":
                    queue.append((position, index + 1, collected, steps + 1))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    target = (row + dr, col + dc)
                    nr, nc = target
                    if not (0 <= nr < 3 and 0 <= nc < 4) or grid[nr][nc] in {"#", "H"}:
                        continue
                    event = {"C": "COIN", "E": "GREEN_KEY", "G": "GOAL"}.get(grid[nr][nc])
                    if target in collected:
                        event = None
                    if event and event != trace[index]:
                        continue
                    next_collected = collected | {target} if event in {"COIN", "GREEN_KEY"} else collected
                    queue.append((target, index + bool(event), frozenset(next_collected), steps + 1))
            return None

        rng = random.Random(42)
        trace = ["COIN", "BELL", "BELL", "COIN", "GREEN_KEY", "BELL", "BELL", "GOAL"]
        valid_cases = 0
        for case in range(20):
            cells = list("CCES#H...")
            rng.shuffle(cells)
            cells = ["A"] + cells + [".", "G"]
            grid = [cells[index:index + 4] for index in range(0, 12, 4)]
            with self.subTest(case=case):
                expected = reference_length(grid, trace)
                route = shortest_valid_trace_events(grid, trace)
                self.assertEqual(len(route) if route is not None else None, expected)
                if route is not None:
                    valid_cases += 1
                    self.assertEqual([event for event in route if event], trace)
        self.assertGreater(valid_cases, 0)

    def test_repeated_bell_task_has_exact_route_and_reward_bounds(self):
        self.spec.update(coins=14, bells=1)
        self.spec["keys"]["green"] = 1
        trace = (["COIN"] * 12 + ["BELL"] * 37 + ["COIN"] * 2
                 + ["GREEN_KEY"] + ["BELL"] * 12 + ["GOAL"])
        env = generate_gridworld(42, json.dumps(self.spec), trace)
        route = shortest_valid_trace_events(env.grid, trace)
        self.assertEqual(len(route), 111)
        self.assertEqual([event for event in route if event], trace)
        crm = SimpleNamespace(
            u_0=0, c_0=(),
            transition=lambda state, counters, props: (0, (), 1.0 if props else -0.1),
            _get_reward_transition_function=lambda: {0: {"/ (-)": -0.1, "GOAL / (-)": 1.0}},
            _get_state_transition_function=lambda: {0: {"GOAL / (-)": -1}},
        )
        bounds = calculate_trace_reward_bounds(crm, trace, 200, env.grid)
        self.assertEqual(bounds["shortest_valid_steps"], 111)
        self.assertAlmostEqual(bounds["maximum_reward"], len(trace) - (111 - len(trace)) * 0.1)


if __name__ == "__main__":
    unittest.main()

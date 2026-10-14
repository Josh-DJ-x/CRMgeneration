"""Regression checks for the three-key generation failure, without LLM calls."""

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system_specifications import GENERATION_FORMAT_SHORT
import system_loop
from system_functions import (
    Event, crm_accepts_trace, crm_parse, crm_trace_alignment_error, extract_generation_trace,
)


TRACE = ("GREEN_KEY", "BLUE_KEY", "RED_KEY", "GOAL")


def ordered_crm():
    sections = ["Initial State\nu0 = 0\nInitial Counters\nc0 = ()"]
    for heading, values in [
        ("State Transitions", [(1, 0), (2, 1), (3, 2), (-1, 3)]),
        ("Counter Transitions", [("()", "()")] * 4),
        ("Reward Transitions", [(1, -0.01)] * 3 + [(10, -0.01)]),
    ]:
        sections.append(heading)
        for state, (event, (advance, default)) in enumerate(zip(TRACE, values)):
            sections.append(f"State {state}\n    {event} / (-) -> {advance}\n    / (-) -> {default}")
    return "\n".join(sections)


class AlignmentTests(unittest.TestCase):
    def test_ordered_keys_need_no_counters_and_tolerate_movement(self):
        crm = crm_parse(ordered_crm())
        self.assertTrue(crm_accepts_trace(crm, TRACE))
        state, counters = crm.u_0, crm.c_0
        for name in TRACE:
            for _ in range(3):
                next_state, next_counters, _ = crm.transition(state, counters, set())
                self.assertEqual((next_state, next_counters), (state, counters))
            state, counters, _ = crm.transition(state, counters, {Event[name]})
        self.assertIn(state, crm.F)

    def test_accepts_full_trace_despite_unused_early_goal_transition(self):
        text = ordered_crm().replace("    GREEN_KEY / (-) -> 1",
                                   "    GREEN_KEY / (-) -> 1\n    GOAL / (-) -> -1")
        text = text.replace("    GREEN_KEY / (-) -> ()",
                            "    GREEN_KEY / (-) -> ()\n    GOAL / (-) -> ()")
        text = text.replace("    GREEN_KEY / (-) -> 1\n    / (-) -> -0.01",
                            "    GREEN_KEY / (-) -> 1\n    GOAL / (-) -> -0.01\n    / (-) -> -0.01")
        crm = crm_parse(text)
        self.assertTrue(crm_accepts_trace(crm, TRACE))
        self.assertIsNone(crm_trace_alignment_error(crm, TRACE))

    def test_failed_run_reports_counter_stall(self):
        text = (Path(__file__).parent / "fixtures" / "three_keys_failed_crm.txt").read_text()
        crm = crm_parse(text)
        error = crm_trace_alignment_error(crm, TRACE)
        self.assertIn("BLUE_KEY: State 1 -> 1", error)
        self.assertIn("counters (0, 1, 1)", error)
        self.assertFalse(crm_accepts_trace(crm, TRACE))

    def test_goal_must_terminate_without_extra_step(self):
        text = ordered_crm().replace("GOAL / (-) -> -1", "GOAL / (-) -> 3")
        self.assertIn("Final GOAL did not reach terminal", crm_trace_alignment_error(crm_parse(text), TRACE))

    def test_short_example_executes_declared_trace(self):
        trace, text, _ = extract_generation_trace(GENERATION_FORMAT_SHORT, initial=True)
        self.assertTrue(crm_accepts_trace(crm_parse(text), trace))

    def test_undefined_event_becomes_replay_diagnostic(self):
        crm = crm_parse(ordered_crm().replace("RED_KEY /", "RED /"))
        self.assertFalse(crm_accepts_trace(crm, TRACE))
        error = crm_trace_alignment_error(crm, TRACE)
        self.assertIn("NameError", error)
        self.assertIn("name 'RED' is not defined", error)
        self.assertIn("RED_KEY", error)
        self.assertIn("BLUE_KEY: State 1 -> 2", error)

    def test_replay_runtime_failures_become_validation_feedback(self):
        for error in (IndexError("bad counter index"), TypeError("invalid formula"),
                      ZeroDivisionError("bad formula arithmetic")):
            with self.subTest(error=type(error).__name__):
                crm = SimpleNamespace(u_0=0, c_0=(), F={-1})
                def transition(*args):
                    raise error
                crm.transition = transition
                self.assertFalse(crm_accepts_trace(crm, TRACE))
                self.assertIn(type(error).__name__, crm_trace_alignment_error(crm, TRACE))

    def test_undefined_event_is_retried_and_can_be_corrected(self):
        broken = ordered_crm().replace("RED_KEY /", "RED /")
        initial = ('<<<DESIRED_TRACE>>>\nGREEN_KEY -> BLUE_KEY -> RED_KEY -> GOAL\n'
                   '<<<ENVIRONMENT>>>\n{"coins":0,"keys":{"green":1,"blue":1,"red":1,"yellow":0},'
                   '"bells":0,"hazards":false}\n<<<CRM>>>\n' + broken)
        for repaired in (False, True):
            with self.subTest(repaired=repaired):
                responses = iter([initial] + ["<<<CRM>>>\n" +
                    (ordered_crm() if repaired else broken)] * 3)
                prompts = []
                def generate(prompt):
                    prompts.append(prompt)
                    return next(responses)
                with (patch("builtins.input", return_value="Collect three ordered keys"),
                      patch.object(system_loop, "configure_tracking", return_value=None),
                      patch.object(system_loop, "log_generation_failure") as log_failure,
                      patch.object(system_loop, "generate_gridworld",
                                   side_effect=RuntimeError("reached environment generation")) as environment):
                    expected = RuntimeError if repaired else SystemExit
                    message = "reached environment generation" if repaired else "after three corrections"
                    with self.assertRaisesRegex(expected, message):
                        system_loop.main(gemini=SimpleNamespace(generate=generate),
                                         openrouter=SimpleNamespace(), long=False, iterative=False)
                self.assertEqual(len(prompts), 2 if repaired else 4)
                for prompt in prompts[1:]:
                    self.assertIn("NameError", prompt)
                    self.assertIn("name 'RED' is not defined", prompt)
                    self.assertIn("GREEN_KEY -> BLUE_KEY -> RED_KEY -> GOAL", prompt)
                    self.assertIn("RED_KEY", prompt)
                self.assertEqual(environment.call_count, 1 if repaired else 0)
                if not repaired:
                    record = log_failure.call_args.args[0]
                    self.assertEqual(record["generator_call_count"], 4)
                    self.assertEqual(record["crm_parse_fail_percentage"], 100)

    def test_counter_dimensions_must_match_initial_configuration(self):
        failed = (Path(__file__).parent / "fixtures" / "three_keys_failed_crm.txt").read_text()
        with self.assertRaisesRegex(ValueError, "condition.*3.*positions"):
            crm_parse(failed.replace("/ (NZ)", "/ (NZ,-,-,-)"))
        with self.assertRaisesRegex(ValueError, "update.*3 positions"):
            crm_parse(failed.replace("-> (-1, 0, 0)", "-> (-1,)"))

    def test_corrections_include_fixed_trace_previous_crm_and_error(self):
        failed = (Path(__file__).parent / "fixtures" / "three_keys_failed_crm.txt").read_text()
        for text, diagnostic in [(failed, "BLUE_KEY: State 1 -> 1"),
                                 (failed.replace("c0 = (1, 1, 1)", "c0 = (1)"), "tuple of integers")]:
            with self.subTest(diagnostic=diagnostic):
                initial = ('<<<DESIRED_TRACE>>>\nGREEN_KEY -> BLUE_KEY -> RED_KEY -> GOAL\n'
                           '<<<ENVIRONMENT>>>\n{"coins":0,"keys":{"green":1,"blue":1,"red":1,"yellow":0},'
                           '"bells":0,"hazards":false}\n<<<CRM>>>\n' + text)
                responses = iter([initial] + ["<<<CRM>>>\n" + text] * 3)
                prompts = []
                def generate(prompt):
                    prompts.append(prompt)
                    return next(responses)
                with (patch("builtins.input", return_value="Collect three ordered keys"),
                      patch.object(system_loop, "configure_tracking", return_value=None),
                      patch.object(system_loop, "log_generation_failure") as log_failure):
                    with self.assertRaisesRegex(SystemExit, "after three corrections"):
                        system_loop.main(gemini=SimpleNamespace(generate=generate),
                                         openrouter=SimpleNamespace(), long=False, iterative=False)
                self.assertEqual(len(prompts), 4)
                for prompt in prompts[1:]:
                    self.assertIn(diagnostic, prompt)
                    self.assertIn("GREEN_KEY -> BLUE_KEY -> RED_KEY -> GOAL", prompt)
                    self.assertIn(text, prompt)
                    self.assertIn("Counter Transitions", prompt)
                self.assertEqual(log_failure.call_args.args[0]["generator_call_count"], 4)


if __name__ == "__main__":
    unittest.main()

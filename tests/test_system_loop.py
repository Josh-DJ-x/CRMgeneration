"""Offline tests for the CRM orchestration loop.

These tests never call Gemini or OpenRouter. ``FakeLLM`` supplies a fixed
generation response, while environment and training functions are replaced
with deterministic test doubles. This isolates parsing and loop-control
behaviour from API cost, stochastic learning, and network access.
"""

import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import system_loop
import system_specifications as prompt_specs
from system_specifications import GENERATION_FORMAT, GENERATION_FORMAT_SHORT
from system_functions import (
    Gemini,
    Groq,
    Ollama,
    OpenRouter,
    OpenRouterResponseError,
    EnvironmentGenerationError,
    RouteSearchLimitError,
    combined_difficulty,
    extract_generation_trace,
    parse_compact_trace,
    environment_difficulty,
    format_compact_trace,
    shortest_valid_trace_events,
)


class FakeLLM:
    """Scripted replacement for an LLM client that records received prompts."""

    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class SystemLoopTests(unittest.TestCase):
    def test_openrouter_rejects_partial_length_limited_answer_without_saving_it(self):
        with patch("system_functions.OpenAI") as factory, \
             patch.dict("os.environ", {"OPENROUTER_MAX_TOKENS": "8192"}):
            create = factory.return_value.chat.completions.create
            create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="<<<REFLECTION>>>unfinished"), finish_reason="length")])
            judge = OpenRouter("test-key")
            with self.assertRaises(OpenRouterResponseError) as raised:
                judge.generate("evaluate")
            self.assertEqual(raised.exception.finish_reason, "length")
            self.assertEqual(create.call_args.kwargs["max_tokens"], 16384)
            self.assertEqual(judge.messages, [])

    def test_openrouter_rejects_missing_choices_and_empty_answers_without_polluting_history(self):
        for response, diagnostic in [
            (SimpleNamespace(choices=None), "no completion choices"),
            (SimpleNamespace(choices=[]), "no completion choices"),
            (SimpleNamespace(choices=[SimpleNamespace(message=None)]), "no answer text"),
            (SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=" "), finish_reason="length")]), "token budget"),
        ]:
            with self.subTest(diagnostic=diagnostic), patch("system_functions.OpenAI") as factory:
                factory.return_value.chat.completions.create.return_value = response
                judge = OpenRouter("test-key")
                with self.assertRaisesRegex(OpenRouterResponseError, diagnostic):
                    judge.generate("evaluate")
                self.assertEqual(judge.messages, [])
                self.assertEqual(judge.resolved_models, [])

    def test_openrouter_surfaces_error_body_and_status(self):
        with patch("system_functions.OpenAI") as factory:
            factory.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=None, error={"code": 503, "message": "Provider unavailable"})
            judge = OpenRouter("test-key")
            with self.assertRaisesRegex(OpenRouterResponseError, "503.*Provider unavailable") as raised:
                judge.generate("evaluate")
            self.assertEqual(raised.exception.status_code, 503)
            self.assertEqual(judge.messages, [])

    def test_openrouter_records_model_for_each_response(self):
        with patch("system_functions.OpenAI") as client_type:
            client = client_type.return_value
            client.chat.completions.create.side_effect = [
                SimpleNamespace(model="vendor/model-a", choices=[
                    SimpleNamespace(message=SimpleNamespace(content="first"))]),
                SimpleNamespace(model="vendor/model-b", choices=[
                    SimpleNamespace(message=SimpleNamespace(content="second"))]),
            ]
            judge = OpenRouter("test-key")
            judge.generate("one")
            judge.generate("two")
        self.assertEqual(judge.model, "openrouter/free")
        self.assertEqual(judge.resolved_models, ["vendor/model-a", "vendor/model-b"])
        self.assertEqual(judge.last_response_model, "vendor/model-b")
        self.assertEqual(client.chat.completions.create.call_args.kwargs["messages"], [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
        ])
        self.assertEqual(judge.messages, [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "second"},
        ])

    def test_judge_followup_contains_latest_data_without_static_instructions(self):
        raw_run = '{"trace": ["GOAL"], "steps": 12}'
        prompt = system_loop._judge_followup_prompt(2, "revised CRM", raw_run, ("GOAL",))
        self.assertIn("Iteration 2", prompt)
        self.assertIn("revised CRM", prompt)
        self.assertIn(raw_run, prompt)
        self.assertIn('["GOAL"]', prompt)
        self.assertIn("<<<REFLECTION>>>", prompt)
        self.assertIn("Compare this revision with the previous iteration", prompt)
        self.assertIn("step_efficiency, and reward_efficiency", prompt)
        self.assertIn("Both efficiency thresholds still apply", prompt)
        self.assertNotIn(prompt_specs.CRM_FORMAT.strip(), prompt)
        self.assertNotIn(prompt_specs.ENVIRONMENT_SEMANTICS.strip(), prompt)

    def setUp(self):
        # Every orchestration test uses fake providers; keep early exits out of
        # the user's real MLflow database, including future test cases.
        failure_log_patch = patch.object(system_loop, "log_generation_failure")
        failure_log_patch.start()
        self.addCleanup(failure_log_patch.stop)

    def test_gemini_model_can_be_selected(self):
        with patch("system_functions.genai.Client") as client_type:
            llm = Gemini("test-key", model="gemini-3.5-flash-lite")
        self.assertEqual(llm.model, "gemini-3.5-flash-lite")
        client_type.return_value.chats.create.assert_called_once_with(
            model="gemini-3.5-flash-lite"
        )

    def test_ollama_keeps_successful_conversation_history(self):
        with patch("system_functions.OpenAI") as client_type:
            client = client_type.return_value
            client.chat.completions.create.side_effect = [
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="one"))]),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="two"))]),
            ]
            llm = Ollama(model="local-model")
            self.assertEqual(llm.generate("first"), "one")
            self.assertEqual(llm.generate("second"), "two")

        client_type.assert_called_once_with(
            base_url="http://localhost:11434/v1/", api_key="ollama"
        )
        self.assertEqual(client.chat.completions.create.call_args.kwargs, {
            "model": "local-model",
            "temperature": 0,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "one"},
                {"role": "user", "content": "second"},
            ],
        })

    def test_groq_uses_api_key_and_keeps_conversation_history(self):
        with patch("system_functions.OpenAI") as client_type:
            client = client_type.return_value
            client.chat.completions.create.side_effect = [
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="one"))]),
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="two"))]),
            ]
            llm = Groq(api_key="test-key", model="test-model")
            self.assertEqual(llm.generate("first"), "one")
            self.assertEqual(llm.generate("second"), "two")

        client_type.assert_called_once_with(
            base_url="https://api.groq.com/openai/v1", api_key="test-key"
        )
        self.assertEqual(client.chat.completions.create.call_args.kwargs, {
            "model": "test-model",
            "temperature": 0,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "one"},
                {"role": "user", "content": "second"},
            ],
        })

    def test_groq_selection_uses_environment_configuration(self):
        with (
            patch("builtins.input", return_value="Reach the goal."),
            patch.object(system_loop, "configure_tracking", return_value=None),
            patch.object(system_loop, "Gemini") as gemini_type,
            patch.object(system_loop, "Groq") as groq_type,
            patch.dict("os.environ", {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "test-model"}),
        ):
            groq_type.return_value.generate.return_value = "<<<INVALID_TASK>>>"
            with self.assertRaisesRegex(SystemExit, "Please specify a valid task"):
                system_loop.main(generator_backend="groq", openrouter=FakeLLM())
        gemini_type.assert_not_called()
        groq_type.assert_called_once_with("test-key", model="test-model")

    def test_ollama_selection_does_not_create_gemini(self):
        with (
            patch("builtins.input", return_value="Reach the goal."),
            patch.object(system_loop, "configure_tracking", return_value=None),
            patch.object(system_loop, "Gemini") as gemini_type,
            patch.object(system_loop, "Ollama") as ollama_type,
            patch.dict("os.environ", {"OLLAMA_MODEL": "my-local-model"}),
        ):
            ollama_type.return_value.generate.return_value = "<<<INVALID_TASK>>>"
            with self.assertRaisesRegex(SystemExit, "Please specify a valid task"):
                system_loop.main(generator_backend="ollama", openrouter=FakeLLM())
        gemini_type.assert_not_called()
        ollama_type.assert_called_once_with(model="my-local-model")

    def test_invalid_compact_trace_gets_three_corrections_in_one_shot_mode(self):
        generation = re.sub(
            r"(<<<DESIRED_TRACE>>>\s*)[^\n]+",
            lambda match: match.group(1) + "COIN x 0 -> GOAL",
            GENERATION_FORMAT,
            count=1,
        )
        generator = FakeLLM([generation] * 4)
        with (
            patch("builtins.input", return_value="Collect five coins."),
            patch.object(system_loop, "configure_tracking", return_value=None),
            patch.object(system_loop, "log_generation_failure") as log_failure,
        ):
            with self.assertRaisesRegex(SystemExit, "positive_count"):
                system_loop.main(gemini=generator, openrouter=FakeLLM(), iterative=False)
        self.assertEqual(len(generator.prompts), 4)
        record = log_failure.call_args.args[0]
        self.assertNotIn("trace_verification_failures", record)
        self.assertEqual(record["generator_call_count"], 4)

    def test_provider_capacity_errors_exit_cleanly(self):
        class ProviderError(Exception):
            def __init__(self, status_code, message):
                super().__init__(message)
                self.status_code = status_code

        for error, expected in (
            (ProviderError(429, "request failed"), "rate limited"),
            (ProviderError(503, "request failed"), "high demand"),
            (ProviderError(500, "model is overloaded"), "high demand"),
        ):
            with self.subTest(error=error):
                generator = FakeLLM([error])
                with (
                    patch("builtins.input", return_value="Reach the goal."),
                    patch.object(system_loop, "configure_tracking", return_value=None),
                    patch.object(system_loop, "log_generation_failure") as log_failure,
                ):
                    with self.assertRaisesRegex(SystemExit, expected):
                        system_loop.main(gemini=generator, openrouter=FakeLLM())
                self.assertEqual(len(generator.prompts), 1)
                self.assertEqual(log_failure.call_args.args[0]["failure_category"],
                                 "rate_limit" if expected == "rate limited" else "provider_capacity")

    def test_unrelated_llm_error_is_not_hidden(self):
        self.assertIsNone(system_loop._llm_capacity_message(ValueError("invalid response")))

    def test_environment_generation_failure_gets_three_repairs_in_one_shot(self):
        with (
            patch("builtins.input", return_value="Collect five coins."),
            patch.object(system_loop, "configure_tracking", return_value=None),
            patch.object(system_loop, "generate_gridworld", side_effect=EnvironmentGenerationError("time limit")),
            patch.object(system_loop, "train_agent") as train,
            patch.object(system_loop, "log_generation_failure") as log_failure,
        ):
            generator = FakeLLM([GENERATION_FORMAT] * 4)
            with self.assertRaisesRegex(SystemExit, "after three corrections.*time limit"):
                system_loop.main(gemini=generator, openrouter=FakeLLM(), iterative=False)
        self.assertEqual(len(generator.prompts), 4)
        train.assert_not_called()
        log_failure.assert_called_once()
        self.assertIn("EnvironmentGenerationError", log_failure.call_args.args[0]["failure_reason"])

    def test_reward_bound_route_failures_exit_cleanly_with_distinct_categories(self):
        for error, category in (
            (ValueError("No ordered route exists for the desired trace."), "no_valid_environment"),
        ):
            with (
                self.subTest(category=category),
                patch("builtins.input", return_value="Collect five coins."),
                patch.object(system_loop, "configure_tracking", return_value=None),
                patch.object(system_loop, "generate_gridworld", return_value=SimpleNamespace(grid=[])),
                patch.object(system_loop, "GridWorldLabellingFunction"),
                patch.object(system_loop, "GridWorldCrossProduct", return_value=SimpleNamespace(max_steps=200, ground_env=None)),
                patch.object(system_loop, "calculate_trace_reward_bounds", side_effect=error),
                patch.object(system_loop, "train_agent") as train,
                patch.object(system_loop, "log_generation_failure") as log_failure,
            ):
                with self.assertRaises(SystemExit) as raised:
                    system_loop.main(gemini=FakeLLM([GENERATION_FORMAT] * 4), openrouter=FakeLLM(), iterative=False)
                train.assert_not_called()
                log_failure.assert_called_once()
                if isinstance(error, RouteSearchLimitError):
                    self.assertEqual(log_failure.call_args.args[0]["failure_category"], category)
                    self.assertEqual(log_failure.call_args.args[0]["failure_stage"], "evaluation reward bounds")
                else:
                    self.assertIn("after three corrections", str(raised.exception))

    def test_invalid_environment_one_shot_gets_corrections_before_logging_failure(self):
        for changes in ({"coins": 100}, {"hazards": "true"}):
            with self.subTest(changes=changes):
                trace, crm, environment = extract_generation_trace(
                    GENERATION_FORMAT, initial=True
                )
                spec = json.loads(environment)
                spec.update(changes)
                generation = GENERATION_FORMAT.replace(environment, json.dumps(spec))
                generator = FakeLLM([generation] * 4)
                judge = FakeLLM()
                with (
                    patch("builtins.input", return_value="Collect five coins."),
                    patch.object(system_loop, "configure_tracking", return_value=None),
                    patch.object(system_loop, "train_agent") as train,
                    patch.object(system_loop, "log_generation_failure") as log_failure,
                ):
                    with self.assertRaisesRegex(
                        SystemExit,
                        "after three corrections.*Invalid environment specification",
                    ) as raised:
                        system_loop.main(gemini=generator, openrouter=judge, iterative=False)
                train.assert_not_called()
                self.assertEqual(len(generator.prompts), 4)
                self.assertEqual(judge.prompts, [])
                log_failure.assert_called_once()
                record = log_failure.call_args.args[0]
                self.assertEqual(record["failure_stage"], "CRM correction validation")
                self.assertIn("after three corrections", record["failure_reason"])
                self.assertEqual(record["generation_history"][0]["response"], generation)

    def test_empty_task_is_rejected_without_llm_calls(self):
        """Whitespace-only input is rejected before any provider is created or called."""
        with patch("builtins.input", return_value="   "):
            with self.assertRaisesRegex(SystemExit, "Please specify a valid task"):
                system_loop.main()

    def test_invalid_task_sentinel_is_rejected(self):
        """The agreed LLM sentinel stops an incoherent task before training."""
        generator = FakeLLM(["<<<INVALID_TASK>>>"])
        with (
            patch("builtins.input", return_value="aubihd ao; hehf ah8e 9a"),
            patch.object(system_loop, "configure_tracking", return_value=None),
        ):
            with self.assertRaisesRegex(SystemExit, "Please specify a valid task"):
                system_loop.main(gemini=generator, openrouter=FakeLLM())
        self.assertEqual(len(generator.prompts), 1)

    def test_initial_prompt_supplies_task_interpretation_in_both_modes(self):
        for use_long in (True, False):
            with self.subTest(long=use_long):
                generator = FakeLLM(["<<<INVALID_TASK>>>"])
                with patch.object(system_loop, "configure_tracking", return_value=None):
                    with self.assertRaises(SystemExit):
                        system_loop.main(
                            gemini=generator, openrouter=FakeLLM(),
                            task_description="Pick up a coin", long=use_long,
                        )
                sent_prompt = generator.prompts[0]
                specifications = prompt_specs.select_specifications(use_long)
                self.assertNotIn("GENERATOR_CONTEXT", specifications)
                self.assertIn(specifications["CRM_GENERATION_PROMPT"].strip(), sent_prompt)
                self.assertIn('"Pick up a coin" is valid', sent_prompt)
                self.assertIn("COIN -> GOAL", sent_prompt)
                self.assertIn("omitted goal is implicit" if not use_long else
                              "An omitted goal is implicit", sent_prompt)
                self.assertIn("This is the FIRST generation, not a refinement", sent_prompt)
                self.assertTrue(sent_prompt.endswith("Task:\n\nPick up a coin"))

    def test_generation_format_parses_nested_environment_json(self):
        """The ``keys`` object does not truncate the environment JSON."""
        trace, crm, environment = extract_generation_trace(
            GENERATION_FORMAT, initial=True
        )

        self.assertEqual(
            trace,
            ("COIN", "COIN", "COIN", "COIN", "COIN", "RED_KEY", "GOAL"),
        )
        self.assertIn("Initial State", crm)
        self.assertEqual(
            json.loads(environment)["keys"],
            {"red": 1, "blue": 0, "green": 0, "yellow": 0},
        )

    def test_short_specifications_are_approximately_40_percent_and_parseable(self):
        token_proxy = lambda value: len(re.findall(r"\w+|[^\w\s]", value))
        # This word/punctuation proxy differs from model tokenizers. Allow five
        # percentage points around the 40% target; exact tokenizer counts
        # are recorded in outputs/short_specifications_token_counts.json.
        long_size = sum(
            token_proxy(getattr(prompt_specs, name))
            for name in prompt_specs._PROMPT_SPECIFICATION_NAMES if name != "TASK_PROMPT"
        )
        short_size = sum(
            token_proxy(getattr(prompt_specs, name + "_SHORT"))
            for name in prompt_specs._PROMPT_SPECIFICATION_NAMES if name != "TASK_PROMPT"
        )
        trace, crm, environment = extract_generation_trace(
            GENERATION_FORMAT_SHORT, initial=True
        )

        self.assertGreaterEqual(short_size / long_size, 0.35)
        self.assertLessEqual(short_size / long_size, 0.45)
        self.assertEqual(prompt_specs.TASK_PROMPT_SHORT, prompt_specs.TASK_PROMPT)
        initial_names = ("CRM_GENERATION_PROMPT", "GENERATION_FORMAT", "ENVIRONMENT_CONTEXT",
                         "ENVIRONMENT_SEMANTICS", "CRM_FORMAT")
        initial_ratio = (
            sum(token_proxy(getattr(prompt_specs, name + "_SHORT")) for name in initial_names)
            / sum(token_proxy(getattr(prompt_specs, name)) for name in initial_names)
        )
        self.assertGreaterEqual(initial_ratio, 0.35)
        self.assertLessEqual(initial_ratio, 0.45)
        self.assertEqual(trace, ("GOAL",))
        self.assertEqual(json.loads(environment)["keys"]["green"], 0)
        self.assertIn("Initial State", crm)

    def test_short_prompts_cover_correction_and_judge_without_final_judge_call(self):
        initial = GENERATION_FORMAT_SHORT.split("<<<CRM>>>")[0] + "<<<CRM>>>\ninvalid CRM"
        refinement = "<<<CRM>>>" + GENERATION_FORMAT_SHORT.split("<<<CRM>>>")[1]
        generator = FakeLLM([initial, refinement, refinement, refinement])
        judge = FakeLLM(["<<<REFLECTION>>>Try again"] * 2)
        ground = SimpleNamespace(grid=SimpleNamespace(tolist=lambda: []))
        run = {"trace": (), "reward_efficiency": 0.0, "episode_reward": 0.0,
               "minimum_reward": 0.0, "maximum_reward": 1.0, "steps": 200,
               "grid_progression": [], "bell_rings": []}
        with (
            patch("builtins.input", return_value="Reach the goal"),
            patch.object(system_loop, "configure_tracking", return_value=None),
            patch.object(system_loop, "generate_gridworld", return_value=ground),
            patch.object(system_loop, "GridWorldLabellingFunction", return_value=object()),
            patch.object(system_loop, "GridWorldCrossProduct", return_value=SimpleNamespace(max_steps=200)),
            patch.object(system_loop, "train_agent", return_value=({}, [])),
            patch.object(system_loop, "calculate_trace_reward_bounds", return_value={"shortest_valid_steps": 1}),
            patch.object(system_loop, "run_agent", return_value=run),
            patch.object(system_loop, "crm_specification", return_value={}),
            patch.object(system_loop, "combined_difficulty", return_value={"difficulty_score": 0.1}),
            patch.object(system_loop, "export_grid_animation", return_value=("test.gif", True)),
            patch.object(system_loop, "log_experiment_run", return_value="test") as log_run,
        ):
            system_loop.main(gemini=generator, openrouter=judge)
        self.assertIn(prompt_specs.CRM_GENERATION_PROMPT_SHORT, generator.prompts[0])
        self.assertIn(prompt_specs.CRM_FORMAT_SHORT, generator.prompts[0])
        self.assertIn(prompt_specs.COMMON_PITFALLS_SHORT, generator.prompts[1])
        for prompt in generator.prompts[2:]:
            self.assertIn(prompt_specs.REFINEMENT_CONTEXT_SHORT, prompt)
        self.assertEqual(len(judge.prompts), 2)
        self.assertIn(prompt_specs.REFLECTION_FORMAT_SHORT, judge.prompts[0])
        self.assertIn(prompt_specs.FEEDBACK_FORMAT_SHORT, judge.prompts[0])
        self.assertIn(prompt_specs.CRM_FORMAT_SHORT, judge.prompts[0])
        self.assertIn("Iteration 2", judge.prompts[1])
        record = log_run.call_args.args[0]
        self.assertFalse(record["long"])
        self.assertEqual(record["iterations"], 3)
        self.assertEqual(record["judge_call_count"], 2)
        self.assertIsNone(record["iteration_history"][-1]["judge_feedback"])

    def test_compact_trace_expands_to_the_exact_declared_trace(self):
        response = "COIN x 12 -> BELL x 40 -> COIN x 4 -> GOAL"
        expected = ("COIN",) * 12 + ("BELL",) * 40 + ("COIN",) * 4 + ("GOAL",)

        self.assertEqual(parse_compact_trace(response), expected)
        self.assertEqual(
            format_compact_trace(expected), response,
        )
        self.assertEqual(parse_compact_trace("GREEN_KEY → BELL × 3 → GOAL"),
                         ("GREEN_KEY", "BELL", "BELL", "BELL", "GOAL"))
        for invalid in ["", "COIN x 0 -> GOAL", "COIN x -1 -> GOAL", "COIN x 1.5 -> GOAL",
                        "UNKNOWN -> GOAL", "GOAL -> COIN", "GOAL x 2", "COIN ->",
                        "COIN x 10001 -> GOAL"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_compact_trace(invalid)

    def test_shortest_route_includes_required_non_event_movement(self):
        """Reward bounds can account for movement and bell interaction steps."""
        grid = [
            ["A", "C", "S", "G"],
            ["#", "#", "#", "#"],
        ]
        route = shortest_valid_trace_events(grid, ("COIN", "BELL", "GOAL"))

        self.assertEqual(route, ("COIN", None, "BELL", "GOAL"))

    def test_environment_difficulty_uses_complete_trace_steps(self):
        grid = [["A", "C", "S", "G"]]
        spec = {
            "coins": 1, "bells": 1,
            "keys": {"red": 0, "blue": 0, "green": 0, "yellow": 0},
        }

        metrics = environment_difficulty(grid, spec, ("COIN", "BELL", "GOAL"))
        simpler_spec = {
            "coins": 0, "bells": 0,
            "keys": {"red": 0, "blue": 0, "green": 0, "yellow": 0},
        }
        simpler = combined_difficulty(
            [["A", ".", ".", "G"]], simpler_spec, desired_trace=("GOAL",)
        )
        harder = combined_difficulty(grid, spec, desired_trace=("COIN", "BELL", "GOAL"))

        self.assertEqual(metrics["minimum_steps_to_complete_trace"], 4)
        self.assertNotIn("shortest_goal_distance", metrics)
        self.assertGreater(harder["difficulty_score"], simpler["difficulty_score"])
        self.assertTrue(0 <= harder["difficulty_score"] <= 1)

    def test_difficulty_calibration_and_repeated_bell_work(self):
        spec = {"coins": 0, "bells": 0,
                "keys": {"red": 1, "blue": 1, "green": 1, "yellow": 0}}
        environment = {"required_object_count": 3, "hazard_count": 0,
                       "minimum_steps_to_complete_trace": 38}
        with patch("system_functions.environment_difficulty", return_value=environment):
            easy = combined_difficulty([], spec, desired_trace=("GREEN_KEY", "BLUE_KEY", "RED_KEY", "GOAL"))
        self.assertEqual(easy["difficulty_score"], 0.18)
        hard_trace = (("GREEN_KEY",) * 3 + ("COIN",) * 12
                      + ("RED_KEY", "BLUE_KEY", "YELLOW_KEY")
                      + ("COIN",) * 7 + ("BELL",) * 23 + ("GOAL",))
        environment.update(required_object_count=26, hazard_count=10,
                           minimum_steps_to_complete_trace=100)
        with patch("system_functions.environment_difficulty", return_value=environment):
            hard = combined_difficulty([], spec, desired_trace=hard_trace)
            fewer_bells = combined_difficulty([], spec, desired_trace=hard_trace[:-24] + ("BELL", "GOAL"))
        self.assertGreaterEqual(hard["difficulty_score"], 0.9)
        self.assertLess(hard["difficulty_score"], 1)
        self.assertGreater(hard["difficulty_score"], fewer_bells["difficulty_score"])

    def test_timing_includes_generation_and_parse_corrections_in_iterations(self):
        clock = [0.0]

        def duration(seconds, value):
            def call(*args, **kwargs):
                clock[0] += seconds
                return value
            return call

        initial = ('<<<DESIRED_TRACE>>>\nGOAL\n<<<ENVIRONMENT>>>\n'
                   '{"coins":0,"bells":0,"keys":{"red":0,"blue":0,"green":0,"yellow":0},"hazards":false}'
                   '\n<<<CRM>>>\ninvalid CRM')
        class TimedGenerator:
            def generate(self, prompt):
                seconds, response = next(self.responses)
                clock[0] += seconds
                return response
        generator = TimedGenerator()
        generator.responses = iter([(3, initial), (5, "<<<CRM>>>corrected"), (7, "<<<CRM>>>refined")])
        judge = SimpleNamespace(generate=duration(11, "<<<REFLECTION>>>Try again"))
        ground = SimpleNamespace(grid=SimpleNamespace(tolist=lambda: []))
        base = {"trace": (), "reward_efficiency": 0.0, "episode_reward": 0.0,
                "minimum_reward": 0.0, "maximum_reward": 1.0,
                "steps": 200, "grid_progression": [], "bell_rings": []}
        evaluations = iter([base, {**base, "trace": ("GOAL",), "reward_efficiency": 1.0}])
        def evaluate(*args, **kwargs):
            return duration(2, next(evaluations))()
        with (
            patch("builtins.input", return_value="Reach the goal"),
            patch.object(system_loop.time, "perf_counter", side_effect=lambda: clock[0]),
            patch.object(system_loop, "configure_tracking", side_effect=duration(2, None)),
            patch.object(system_loop, "crm_parse", side_effect=[None, object(), object()]),
            patch.object(system_loop, "crm_accepts_trace", return_value=True),
            patch.object(system_loop, "generate_gridworld", side_effect=duration(1, ground)),
            patch.object(system_loop, "GridWorldLabellingFunction", return_value=object()),
            patch.object(system_loop, "GridWorldCrossProduct", return_value=SimpleNamespace(max_steps=200)),
            patch.object(system_loop, "train_agent", side_effect=duration(10, ({}, []))),
            patch.object(system_loop, "calculate_trace_reward_bounds", return_value={"shortest_valid_steps": 200}),
            patch.object(system_loop, "run_agent", side_effect=evaluate),
            patch.object(system_loop, "crm_specification", return_value={}),
            patch.object(system_loop, "combined_difficulty", side_effect=duration(1, {"difficulty_score": 0.1})),
            patch.object(system_loop, "export_grid_animation", side_effect=duration(4, ("test.gif", True))),
            patch.object(system_loop, "log_experiment_run", return_value="test") as log_run,
        ):
            system_loop.main(gemini=generator, openrouter=judge)
        record = log_run.call_args.args[0]
        self.assertEqual(record["generator_call_count"], 3)
        self.assertAlmostEqual(record["crm_parse_fail_percentage"], 100 / 3)
        self.assertEqual(record["average_generation_time"], 5)
        self.assertEqual(record["average_iteration_time"], 26)
        self.assertEqual([entry["time_seconds"] for entry in record["iteration_history"]], [32, 20])
        self.assertEqual(record["total_system_time"], 54)
        self.assertEqual(sum(record["timing_breakdown"].values()), 54)
        self.assertEqual(record["timing_breakdown"]["misc_time_seconds"], 6)
        self.assertEqual(record["timing_version"], "task-outcome-v3")
        self.assertEqual(len(record["timing_breakdown"]), 5)

    def test_main_accepts_injected_clients_and_skips_judge_on_success(self):
        """A matching agent trace succeeds without an LLM judge call."""
        generation = """<<<DESIRED_TRACE>>>
GOAL
<<<ENVIRONMENT>>>
{"coins": 0, "keys": {"red": 0, "blue": 0, "green": 0, "yellow": 0}, "bells": 0, "hazards": false}
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
    GOAL / (-) -> 1.0
    / (-) -> 0.0
"""
        generator = FakeLLM([generation])
        one_shot_generator = FakeLLM([generation])
        judge = FakeLLM()

        class DummyEnvironment:
            max_steps = 200

        class DummyGroundEnvironment:
            class Grid:
                @staticmethod
                def tolist():
                    return []

            grid = Grid()

        with (
            patch("builtins.input", return_value="Reach the goal."),
            patch.object(system_loop, "generate_gridworld", return_value=DummyGroundEnvironment()),
            patch.object(system_loop, "GridWorldLabellingFunction", return_value=object()),
            patch.object(system_loop, "GridWorldCrossProduct", return_value=DummyEnvironment()),
            patch.object(system_loop, "train_agent", return_value=({}, [])),
            patch.object(system_loop, "calculate_trace_reward_bounds", return_value={"shortest_valid_steps": 1}),
            patch.object(system_loop, "combined_difficulty", return_value={"difficulty_score": 0.1}),
            patch.object(system_loop, "crm_specification", return_value={}),
            patch.object(system_loop, "export_grid_animation", return_value=("test.gif", True)),
            patch.object(system_loop, "configure_tracking", return_value=None),
            patch.object(system_loop, "log_experiment_run", return_value="fake-run-id") as log_run,
            patch.object(
                system_loop,
                "run_agent",
                side_effect=[
                    {
                        "trace": ("GOAL",), "reward_efficiency": 1.0,
                        "episode_reward": 1.0, "minimum_reward": 0.0,
                        "maximum_reward": 1.0, "steps": 1,
                        "grid_progression": [], "bell_rings": [],
                    },
                    {
                        "trace": (), "reward_efficiency": 0.0,
                        "episode_reward": 0.0, "minimum_reward": 0.0,
                        "maximum_reward": 1.0, "steps": 200,
                        "grid_progression": [], "bell_rings": [],
                    },
                ],
            ),
        ):
            system_loop.main(gemini=generator, openrouter=judge)
            system_loop.main(
                gemini=one_shot_generator, openrouter=judge,
                iterative=False, long=False,
            )

        self.assertEqual(len(generator.prompts), 1)
        self.assertEqual(len(one_shot_generator.prompts), 1)
        self.assertIn("<<<INVALID_TASK>>>", one_shot_generator.prompts[0])
        self.assertNotIn("### Task Validity Gate", one_shot_generator.prompts[0])
        self.assertEqual(judge.prompts, [])
        successful_history = log_run.call_args_list[0].args[0]["iteration_history"]
        self.assertEqual(len(successful_history), 1)
        self.assertEqual(successful_history[0]["agent_trace"], ["GOAL"])
        self.assertTrue(successful_history[0]["trace_correct"])
        self.assertIsNone(successful_history[0]["judge_feedback"])
        failed_history = log_run.call_args_list[1].args[0]["iteration_history"]
        self.assertEqual(failed_history[0]["run_information"]["steps"], 200)
        self.assertFalse(failed_history[0]["trace_correct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

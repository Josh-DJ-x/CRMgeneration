"""Offline provider selection, conversation state, and CLI regression tests."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import system
import system_loop
from crm_support.experiment_results import judge_label
from system_functions import OpenAILLM, OpenRouter, Ollama, Groq, Gemini


class ModelProviderTests(unittest.TestCase):
    def test_all_providers_clear_conversation_between_tasks(self):
        with patch("system_functions.OpenAI"), patch("system_functions.genai.Client"):
            for provider in (OpenAILLM("test-key"), OpenRouter("test-key"),
                             Ollama(), Groq("test-key")):
                with self.subTest(provider=type(provider).__name__):
                    provider.messages = [{"role": "user", "content": "previous task"}]
                    provider.resolved_models = ["old-model"]
                    provider.last_response_model = "old-model"
                    provider.reset_history()
                    self.assertEqual(provider.messages, [])
                    self.assertEqual(provider.resolved_models, [])
                    self.assertIsNone(provider.last_response_model)
            provider = Gemini("test-key")
            provider.client.chats.create.reset_mock()
            provider.reset_history()
            provider.client.chats.create.assert_called_once_with(model=provider.model)

    def test_reused_instances_reset_once_before_each_task(self):
        class ReusedProvider:
            model = "test-model"
            def __init__(self):
                self.messages = ["old task"]
                self.resets = 0
            def reset_history(self):
                self.messages = []
                self.resets += 1
            def generate(self, prompt):
                if self.messages:
                    raise AssertionError("Previous task context leaked")
                self.messages.append(prompt)
                return "<<<INVALID_TASK>>>"
        generator, judge = ReusedProvider(), ReusedProvider()
        with patch.object(system_loop, "configure_tracking", return_value=None), \
             patch.object(system_loop, "log_generation_failure"):
            for task in ("Task one", "Task two"):
                with self.assertRaises(SystemExit):
                    system_loop.main(gemini=generator, openrouter=judge, task_description=task)
        self.assertEqual(generator.resets, 2)
        self.assertEqual(judge.resets, 2)
        self.assertEqual(judge.messages, [])

    def test_openai_preserves_response_items_and_records_actual_model(self):
        output = [{"role": "assistant", "content": "first"}]
        with patch("system_functions.OpenAI") as factory:
            create = factory.return_value.responses.create
            create.side_effect = [
                SimpleNamespace(status="completed", output_text="first", output=output, model="snapshot"),
                SimpleNamespace(status="completed", output_text="second", output=[], model="snapshot"),
            ]
            provider = OpenAILLM("test-key", "test-model")
            self.assertEqual(provider.generate("one"), "first")
            self.assertEqual(provider.generate("two"), "second")
            self.assertEqual(create.call_args.kwargs["input"], [
                {"role": "user", "content": "one"}, *output,
                {"role": "user", "content": "two"},
            ])
            self.assertFalse(create.call_args.kwargs["store"])
            self.assertEqual(provider.resolved_models, ["snapshot", "snapshot"])
            provider.reset_history()
            self.assertEqual(provider.messages, [])
        factory.assert_called_once_with(api_key="test-key")

    def test_openai_rejects_incomplete_or_empty_output_without_changing_history(self):
        for response in [
            SimpleNamespace(status="incomplete", incomplete_details="max_output_tokens"),
            SimpleNamespace(status="completed", output_text=""),
        ]:
            with self.subTest(response=response), patch("system_functions.OpenAI") as factory:
                factory.return_value.responses.create.return_value = response
                provider = OpenAILLM("test-key")
                with self.assertRaises(RuntimeError):
                    provider.generate("evaluate")
                self.assertEqual(provider.messages, [])
                self.assertEqual(provider.resolved_models, [])
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            OpenAILLM(None)

    def test_explicit_generator_model_overrides_environment(self):
        for backend, factory_name, expected_args in [
            ("ollama", "Ollama", ()),
            ("openai", "OpenAILLM", ("test-key",)),
            ("openrouter", "OpenRouter", ("test-router-key",)),
        ]:
            with self.subTest(backend=backend), \
                 patch.object(system_loop, "configure_tracking", return_value=None), \
                 patch.object(system_loop, "log_generation_failure") as logged, \
                 patch.object(system_loop, factory_name) as factory, \
                 patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENROUTER_API_KEY": "test-router-key"}):
                factory.return_value.generate.return_value = "<<<INVALID_TASK>>>"
                with self.assertRaises(SystemExit):
                    system_loop.main(generator_backend=backend, generator_model="chosen-model",
                                     task_description="Reach the goal", openrouter=SimpleNamespace())
                factory.assert_called_once_with(*expected_args, model="chosen-model")
                self.assertEqual(logged.call_args.args[0]["generator_backend"], backend)

    def test_direct_openai_judge_selection(self):
        with patch.object(system_loop, "configure_tracking", return_value=None), \
             patch.object(system_loop, "log_generation_failure"), \
             patch.object(system_loop, "OpenAILLM") as factory, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with self.assertRaises(SystemExit):
                system_loop.main(gemini=SimpleNamespace(generate=lambda _: "<<<INVALID_TASK>>>"),
                                 judge_backend="openai", judge_model="judge-model",
                                 task_description="Reach the goal")
            factory.assert_called_once_with("test-key", model="judge-model")
        self.assertEqual(judge_label("judge-model", ["snapshot"], 1, "openai"), "OpenAI (snapshot)")

    def test_cli_selects_models_and_reuses_supplied_task(self):
        with patch.object(system, "main") as main, patch("builtins.input") as task_input:
            system.cli(["--generator", "ollama", "--generator-model", "qwen2.5:3b",
                        "--judge-provider", "openai", "--judge-model", "gpt-4.1",
                        "--task", "Reach the goal"])
            task_input.assert_not_called()
            self.assertEqual(main.call_args.kwargs["generator_model"], "qwen2.5:3b")
            self.assertEqual(main.call_args.kwargs["judge_backend"], "openai")
            self.assertEqual(main.call_args.kwargs["task_description"], "Reach the goal")


if __name__ == "__main__":
    unittest.main()

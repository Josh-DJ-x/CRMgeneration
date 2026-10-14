"""Offline checks for persistent web jobs and the studio UI."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crm_support.fancy_jobs import JobStore, progress
from scripts.fancy_run import run

class FancyTests(unittest.TestCase):
    def test_jobs_are_serialized_and_survive_reload(self):
        with tempfile.TemporaryDirectory() as temp, patch('crm_support.fancy_jobs.subprocess.Popen') as factory:
            factory.return_value.poll.return_value = None
            store = JobStore(temp)
            job_id = store.submit('Collect two coins')
            self.assertEqual(store.snapshot()[0]['status'], 'running')
            with self.assertRaisesRegex(ValueError, 'already running'):
                store.submit('Reach the goal')
            (Path(temp) / job_id / 'result.json').write_text(json.dumps({'status': 'completed', 'record': {}}))
            factory.return_value.poll.return_value = 0
            self.assertEqual(store.snapshot()[0]['status'], 'completed')
            self.assertEqual(JobStore(temp).snapshot()[0]['task'], 'Collect two coins')

    def test_empty_task_and_interrupted_worker(self):
        with tempfile.TemporaryDirectory() as temp, patch('crm_support.fancy_jobs.subprocess.Popen') as factory:
            store = JobStore(temp)
            with self.assertRaises(ValueError):
                store.submit('  ')
            factory.assert_not_called()
            factory.return_value.poll.return_value = 1
            store.submit('Reach the goal')
            self.assertEqual(store.snapshot()[0]['status'], 'interrupted')

    def test_worker_pins_defaults_and_captures_failure(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ), patch('system_loop.main') as main:
            directory = Path(temp)
            (directory / 'task.txt').write_text('Reach the goal')
            main.return_value = {'success_status': True}
            run(directory)
            self.assertTrue(main.call_args.kwargs['iterative'])
            self.assertTrue(main.call_args.kwargs['long'])
            self.assertEqual(main.call_args.kwargs['generator_model'], 'gemini-3.5-flash-lite')
            self.assertEqual(Path(os.environ['GRIDWORLD_ANIMATION_PATH']).parent, directory)
            self.assertEqual(json.loads((directory / 'result.json').read_text())['status'], 'completed')
            main.side_effect = SystemExit('Provider unavailable')
            run(directory)
            self.assertEqual(json.loads((directory / 'result.json').read_text())['error'], 'Provider unavailable')

    def test_progress_uses_latest_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'run.log').write_text('Generating...\nnoise\nTraining checkpoint 100/2500\n')
            self.assertEqual(progress(temp), 'Training checkpoint 100/2500')

    def test_page_and_empty_submission_without_model_calls(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        st.cache_resource.clear()
        with tempfile.TemporaryDirectory() as temp, patch('crm_support.fancy_jobs.JobStore', return_value=JobStore(temp)):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'web/fancy_app.py'), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.text_area), 1)
            app.text_area[0].input('  ')
            next(button for button in app.button if button.label == 'Run task').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn('Describe a task', app.error[0].value)
        st.cache_resource.clear()

    def test_page_renders_result_and_error_history(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        st.cache_resource.clear()
        with tempfile.TemporaryDirectory() as temp:
            store = JobStore(temp)
            for index, result in enumerate([
                {'status': 'error', 'error': 'Provider unavailable'},
                {'status': 'completed', 'record': {'success_status': True, 'iterations': 1,
                 'total_system_time': 1.2, 'steps_taken': 2,
                 'iteration_history': [{'iteration': 1, 'grid': [['A', '.', 'G']],
                                        'desired_trace': ['GOAL'], 'agent_trace': ['GOAL'],
                                        'environment_spec': {'coins': 0},
                                        'crm_specification': {'u0': 0, 'c0': []}}]}},
            ]):
                directory = Path(temp) / str(index)
                directory.mkdir()
                (directory / 'job.json').write_text(json.dumps({'id': str(index), 'task': 'Reach the goal',
                    'created': f'2026-10-14T12:00:0{index}+00:00'}))
                (directory / 'result.json').write_text(json.dumps(result))
            with patch('crm_support.fancy_jobs.JobStore', return_value=store):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'web/fancy_app.py'), default_timeout=30).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.error[0].value, 'Provider unavailable')
                self.assertEqual(len(app.json), 2)
                self.assertFalse(any(exp.label == 'Iteration results' for exp in app.expander))
                self.assertTrue(any(exp.label == 'Environment specification and generated CRM' for exp in app.expander))
        st.cache_resource.clear()

if __name__ == '__main__':
    unittest.main()

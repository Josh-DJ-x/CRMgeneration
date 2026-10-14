"""GridWorld Studio UI. Launch through system_fancy.py."""

import base64
from datetime import datetime
import html
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
from PIL import Image
import streamlit as st

from crm_support.fancy_jobs import JobStore, progress
from grid_visualizer import render_frame
from system_specifications import TASK_PROMPT

st.set_page_config(page_title="Iterative CRM Generation and Refinement", page_icon="🌿", layout="wide")
st.html("<style>" + (ROOT / "web" / "fancy.css").read_text(encoding="utf-8") + "</style>")


@st.cache_resource
def store():
    return JobStore()


@st.cache_data
def grid_png(rows):
    pygame.font.init()
    surface = render_frame("\n".join(" ".join(row) for row in rows))
    image = Image.frombytes("RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def picture(grid):
    st.image(grid_png(tuple(tuple(row) for row in grid)), width="stretch")


def legend():
    objects = [
        ("A", "Agent", "Carries out your task automatically."),
        ("R", "Red key", 'For example: "collect 2 red keys".'),
        ("B", "Blue key", 'For example: "pick up a blue key".'),
        ("E", "Green key", 'For example: "collect a green key after the coins".'),
        ("Y", "Yellow key", 'For example: "collect 3 yellow keys".'),
        ("C", "Coin", 'Specify a count: "collect 5 coins".'),
        ("S", "Bell", 'Specify how often: "ring the bell 3 times".'),
        ("H", "Hazard", 'Ask to "avoid hazards" to include them.'),
        ("G", "Goal", "Every task ends here, even if you leave it out."),
        ("#", "Wall", "Blocks movement. The system places walls for you."),
    ]
    rows = []
    for symbol, name, detail in objects:
        icon = base64.b64encode(grid_png(((symbol,),))).decode()
        rows.append(f'<tr><td><img src="data:image/png;base64,{icon}" alt="{name}" width="42" height="42"></td>'
                    f'<td><strong>{name}</strong><span>{html.escape(detail)}</span></td></tr>')
    st.html('<table class="legend"><caption>What you can include</caption><tbody>' + ''.join(rows) + '</tbody></table>')


def task_label(task):
    st.html('<div class="eyebrow task-label">TASK DESCRIPTION</div>')
    st.write(task)


def show_job(job, latest=False):
    record = job.get("record") or {}
    status = job["status"]
    successful = record.get("success_status", False)
    label = ("Task complete" if successful else "Task not completed") if status == "completed" else status.title()
    color = "success" if successful else ("pending" if status == "running" else "neutral")
    with st.container(border=True):
        timestamp = datetime.fromisoformat(job["created"]).astimezone().strftime("%d %b · %H:%M")
        st.html(f'<div class="run-top"><span class="eyebrow">{"LATEST TASK" if latest else "PREVIOUS TASK"}</span>'
                f'<span class="badge {color}">{html.escape(label)}</span></div>')
        st.caption(timestamp)
        animation = Path(job["directory"]) / "animation.gif"
        iterations = record.get("iteration_history", [])
        preview = Path(job["directory"]) / "grid.json"
        if animation.is_file() and status == "completed":
            st.image(animation.read_bytes(), width="stretch")
        elif preview.is_file():
            try:
                picture(json.loads(preview.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                st.info("Preparing the grid…")
        elif iterations and iterations[-1].get("grid"):
            picture(iterations[-1]["grid"])
        else:
            st.html('<div class="waiting-grid"><span>✦</span><p>Waiting for grid generation.</p>'
                    '<small>The generated grid will appear here.</small></div>')
        task_label(job["task"])
        if status == "error":
            st.error(job.get("error", "The experiment could not finish."))
        elif status == "interrupted":
            st.warning("This run has no final result. The server or worker may have stopped; you can submit the task again.")
        elif status == "completed":
            seconds = record.get("total_system_time", 0)
            st.caption(f'{record.get("iterations", 0)} iterations  ·  {record.get("steps_taken", "—")} steps  ·  {seconds:.1f}s')
            if record.get("failure_reason"):
                st.warning("The agent did not meet all requirements in this run. Try fewer objects or run the task again.")
            specifications = record.get("specifications") or {}
            final_iteration = iterations[-1] if iterations else {}
            environment = specifications.get("environment_spec") or final_iteration.get("environment_spec")
            crm = specifications.get("crm_specification") or final_iteration.get("crm_specification")
            with st.expander("Environment specification and generated CRM"):
                st.markdown("**Environment specification**")
                if environment is not None:
                    st.json(environment, expanded=True)
                else:
                    st.caption("No environment specification was saved for this run.")
                st.markdown("**Generated CRM**")
                if crm is not None:
                    st.json(crm, expanded=True)
                else:
                    st.caption("No CRM was saved for this run.")



jobs = store().snapshot()
active = next((job for job in jobs if job["status"] == "running"), None)
st.html('<div class="masthead"><div class="brand"><span class="brandmark">&#9638;</span> JOSH SNYMAN</div>'
        '<span class="local-label">HONOURS RESEARCH</span></div>')

left, right = st.columns([1.65, 1], gap="large")
with left:
    st.html('<div class="intro"><h1>Iterative CRM Generation<br>and Refinement</h1>'
            '<p>Generation and evaluation of Counting Reward Machines from natural-language task descriptions.</p>'
            '<div class="research-question"><div class="eyebrow">RESEARCH QUESTION</div>'
            '<p>What prompt-engineering and iterative refinement strategies enable Large Language Models to reliably '
            'generate semantically correct Counting Reward Machines from natural language task descriptions?</p></div></div>')
    with st.container(border=True, key="composer"):
        st.html('<div class="eyebrow">01 / DESCRIBE A TASK</div>')
        st.caption("Choose an example to edit, or write your own task.")
        def fill_example(text):
            st.session_state["task_description"] = text
        examples = [
            ("Collect coins", "Collect 3 coins, then reach the goal."),
            ("Keys and a bell", "Pick up a blue key, then ring the bell twice."),
            ("Avoid hazards", "Collect 2 coins, then reach the goal while avoiding hazards."),
        ]
        for column, (label, example) in zip(st.columns(3), examples):
            column.button(label, on_click=fill_example, args=(example,),
                          disabled=active is not None, width="stretch")
        with st.form("task-form", clear_on_submit=True):
            task = st.text_area("What should the agent do?", key="task_description", height=115, max_chars=4000,
                                placeholder="Collect 3 coins, pick up the blue key, then ring the bell twice.",
                                disabled=active is not None)
            st.caption("The agent learns automatically after you submit your task.")
            submitted = st.form_submit_button("Run task", type="primary", width="stretch",
                                             disabled=active is not None)
        if submitted:
            try:
                store().submit(task)
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(str(error))
    st.html('<div class="section-label"><span class="eyebrow">02 / AGENT REPLAY</span>'
            '<span>Grid and task result</span></div>')
    if active:
        @st.fragment(run_every=2)
        def live_run():
            current = next((job for job in store().snapshot() if job["id"] == active["id"]), None)
            if current is None or current["status"] != "running":
                st.rerun()
            st.info(progress(current["directory"]), icon="⏳")
            show_job(current, latest=True)
        live_run()
    elif jobs:
        show_job(jobs[0], latest=True)
    else:
        with st.container(border=True):
            picture([list(row) for row in (
                "A.........", "..C..#....", ".....#..B.", "...R......", ".......C..",
                ".##.......", "....S.....", ".......H..", "..E..Y....", ".........G")])
            st.caption("Example layout · the generated task environment will replace this preview")
            task_label("No task has been submitted.")
    if len(jobs) > 1:
        st.html('<div class="section-label"><span class="eyebrow">PREVIOUS TASKS</span>'
                f'<span>{len(jobs) - 1} saved locally</span></div>')
        for job in jobs[1:]:
            show_job(job)

with right:
    with st.container(border=True, key="guide"):
        st.html('<div class="eyebrow">HOW TO USE THE PLAYGROUND</div><h2>Write a task for the agent</h2>')
        st.markdown("Tell the agent **what to collect, how many, and in what order**. "
                    "The system creates a new 10 \u00d7 10 grid for your task.")
        legend()
        st.markdown("**A valid task:**\n"
                    "- Uses the objects shown above.\n"
                    "- Gives whole-number counts and a clear order.\n"
                    "- Describes an objective, rather than positions or movement directions.\n"
                    "- Uses small counts so the objects fit in the grid.")
        st.caption("Each key and coin can be collected once. The same bell can be rung repeatedly.")
        st.html('<div class="guide-tip"><b>Example</b><br>'
                '&ldquo;Collect 3 coins, then a blue key, then ring the bell twice.&rdquo;</div>')
        with st.expander("More about valid tasks", expanded=False):
            prompt = TASK_PROMPT[TASK_PROMPT.index("Describe a clear"):]
            prompt = prompt.split("Please enter the task")[0].strip()
            st.markdown(prompt)
st.html('<footer>Josh Snyman <span>Honours Research &middot; Iterative CRM Generation and Refinement</span></footer>')

<<<AGENT_TRACE>>>
["RED_KEY", "BLUE_KEY", "COIN", "COIN", "COIN", "COIN", "COIN", "GOAL"]

<<<RUN_INFORMATION>>>
{...copy the supplied raw run-information JSON exactly...}

<<<REFLECTION>>>
Copy the event trace supplied from the agent run exactly as a JSON array. Then explain
any difference between that trace and the generator's `<<<DESIRED_TRACE>>>`, including
missing events, unexpected events, or an incorrect order. Suggest a CRM revision only
when the traces differ.
Include additional reflection on the agent's performance based on the information from it's most recent run
Include the raw run information supplied by the system in your reflection; do not
alter the `<<<AGENT_TRACE>>>` JSON array or the `<<<RUN_INFORMATION>>>` JSON object.

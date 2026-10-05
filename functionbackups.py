#Requierements implementation

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
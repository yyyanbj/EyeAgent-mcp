#!/usr/bin/env python3
"""This helper script has been removed in favor of the unified CLI.

Use one of the following instead:

  eyeagent run --backend profile --profile triage --patient '{"symptom_text":"..."}'

or call the SymptomTriageAgent via the configured workflow.
"""
def main():  # pragma: no cover
    raise SystemExit(
        "[REMOVED] Use 'eyeagent run' with the triage profile, or run the SymptomTriageAgent in the workflow."
    )


if __name__ == "__main__":  # pragma: no cover
    main()

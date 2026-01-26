"""Prompt template variable documentation."""

TEMPLATE_VARIABLES = {
    "system": ["{philosopher_name}", "{num_philosophers}"],
    "decision": [
        "{philosopher_name}",
        "{state}",
        "{meals_eaten}",
        "{left_fork_status}",
        "{right_fork_status}",
        "{holding_status}",
        "{left_message}",
        "{right_message}",
    ],
}


def get_template_variables() -> dict:
    """Return available template variables."""
    return TEMPLATE_VARIABLES

"""Prompt format shared by training and evaluation. Keep these in sync."""

TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input. "
    "Write a response that completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def build_prompt(instruction: str, input_text: str = "") -> str:
    if input_text and input_text.strip():
        return TEMPLATE_WITH_INPUT.format(instruction=instruction, input=input_text)
    return TEMPLATE.format(instruction=instruction)

import re

def remove_think_blocks(text: str) -> str:
    """
    Removes all occurrences of <think>...</think> (including the tags and content in between)
    from the input multi-line string.
    For Qwen as-per their recommendations: https://huggingface.co/Qwen/Qwen3-235B-A22B
    """
    if text is None:
        return text

    # re.DOTALL makes '.' match newlines as well
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL)

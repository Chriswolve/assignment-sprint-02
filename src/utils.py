

def padLeft(text: str, fixed: int = 0 ,max: int =40) -> str:
    """
    Pads the string with spaces on the right to make it of the specified length.
    """
    return f"{text.rjust(50 - fixed , " ")}" 
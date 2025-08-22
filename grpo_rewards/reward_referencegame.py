import re

def reward_referencegame(completion: str) -> float:
    """
    Reward function for the referencegame format compliance. Only checks the instruction following attribute.

    Args:
        completion: Player's completion string
1
    Returns:
        Float reward: 1.0 if completion follows format, else 0.0
    """
    print(f"\n--------------------------------")
    print(f"completion: {completion}")
    print(f"--------------------------------\n")

    # Check if the completion starts correctly
    if re.match(r"^Expression:\s+.+", completion.strip()):
        return 1.0
    return 0.0

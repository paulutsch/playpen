import re


def reward_taboo(completion: str, prefix: list[dict[str, str]]) -> float:
    """
    Reward function for the Taboo word guessing game.

    Args:
        completion: Single completion string
        prefix: List of dicts with 'content' and 'role' keys containing the conversation

    Returns:
        Float reward in range [0, 1]
    """
    # Extract target word and related words from the initial message
    target_word = None
    related_words = []

    prompt = prefix[0].get("content", "")
    print(f"Extracted prompt: {prompt[:200]}...")

    target_match = re.search(
        r"This is the target word that you need to describe and that the other player needs to guess:\s*\n\s*(\w+)",
        prompt,
        re.DOTALL | re.IGNORECASE,
    )
    if target_match:
        target_word = target_match.group(1).lower().strip()
        print(f"Extracted target word: {target_word}")

    related_section = re.search(
        r"Related words are:\s*\n(.*?)(?:\n\n|Important:)", prompt, re.DOTALL
    )
    if related_section:
        related_text = related_section.group(1)
        related_words = [
            word.strip().lower() for word in re.findall(r"-\s*(\w+)", related_text)
        ]
        print(f"Extracted related words: {related_words}")

    if not re.match(r"^CLUE:\s+.+", completion.strip(), re.IGNORECASE):
        print("Format compliance check failed")
        return 0.0

    reward = 0.5
    print(f"Format compliance passed, base reward: {reward}")

    completion_lower = completion.lower()
    print(f"Completion converted to lowercase: {completion_lower[:100]}...")

    if target_word and target_word not in completion_lower:
        reward += 0.3
        print(f"Target word '{target_word}' not used, bonus added, reward: {reward}")
    else:
        print(f"Target word '{target_word}' was used or not found, no bonus")

    if related_words and not any(
        related_word in completion_lower for related_word in related_words
    ):
        reward += 0.2
        print(f"No related words used, bonus added, reward: {reward}")
    else:
        print(f"Related words were used or not found, no bonus")

    print(f"Final reward: {reward}")
    return reward

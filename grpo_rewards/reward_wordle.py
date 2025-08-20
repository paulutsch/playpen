import re


def reward_wordle(completion: str, prefix: list[dict[str, str]]) -> float:
    """
    Reward function for the Wordle word guessing game.

    Args:
        completion: Single completion string containing the guess and explanation
        prefix: list of messages, format [{content: str, role: str}]

    Returns:
        Float reward in range [0, 1]
    """
    # Extract feedback from the last message (what the completion is responding to)
    feedback = None
    if prefix:
        last_message = prefix[-1].get("content", "")
        feedback_match = re.search(
            r"guess_feedback:\s*([^\n]+)", last_message, re.IGNORECASE
        )
        if feedback_match:
            feedback = feedback_match.group(1).strip()
            print(f"Extracted feedback: {feedback}")

    if not re.match(
        r"^guess:\s*\w+\s*\nexplanation:\s*.+",
        completion.strip(),
        re.IGNORECASE | re.DOTALL,
    ):
        print("Format compliance check failed")
        return 0.0

    reward = 0.5
    print(f"Format compliance passed, base reward: {reward}")

    guess_match = re.search(r"guess:\s*(\w+)", completion, re.IGNORECASE)
    if not guess_match:
        print("No valid guess found")
        return 0.0

    guess = guess_match.group(1).lower().strip()
    print(f"Extracted guess: {guess}")

    if len(guess) == 5 and guess.isalpha():
        reward += 0.2
        print(f"5-letter word bonus added, reward: {reward}")
    else:
        print(f"Not a 5-letter word, returning current reward: {reward}")
        return reward

    if feedback:
        adherence_score = calculate_feedback_adherence(guess, feedback)
        reward += adherence_score
        print(f"Feedback adherence score: {adherence_score}, total reward: {reward}")
    else:
        reward += 0.3
        print(f"No feedback, full credit added, total reward: {reward}")

    print(f"Final reward: {reward}")
    return reward


def calculate_feedback_adherence(guess: str, feedback: str) -> float:
    print(
        f"calculate_feedback_adherence called with guess: {guess}, feedback: {feedback}"
    )

    if len(guess) != 5:
        print("Guess is not 5 letters, returning 0")
        return 0.0

    feedback_parts = feedback.split()
    if len(feedback_parts) != 5:
        print(f"Feedback has {len(feedback_parts)} parts, expected 5, returning 0")
        return 0.0

    adherence_score = 0.0

    for i, part in enumerate(feedback_parts):
        if i >= len(guess):
            break

        match = re.match(r"(\w)<(\w+)>", part)
        if not match:
            print(f"Could not parse feedback part: {part}")
            continue

        feedback_letter = match.group(1).lower()
        status = match.group(2).lower()
        guess_letter = guess[i].lower()

        print(
            f"Position {i}: feedback_letter={feedback_letter}, status={status}, guess_letter={guess_letter}"
        )

        if status == "green":
            if guess_letter == feedback_letter:
                adherence_score += 0.06
                print(
                    f"Green match at position {i}, adherence_score: {adherence_score}"
                )
        elif status == "yellow":
            if guess_letter != feedback_letter and feedback_letter in guess:
                adherence_score += 0.06
                print(
                    f"Yellow match at position {i}, adherence_score: {adherence_score}"
                )
        elif status == "red":
            if guess_letter != feedback_letter and feedback_letter not in guess:
                adherence_score += 0.06
                print(f"Red match at position {i}, adherence_score: {adherence_score}")

    print(f"Final adherence_score: {adherence_score}")
    return adherence_score

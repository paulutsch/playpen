import re


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


def reward_wordle_withcritic(completion: str, prefix: list[dict[str, str]]) -> float:
    """
    Reward function for the Wordle-with-critic task.

    The assistant must respond strictly in lowercase with the format:
        agreement: yes|no
        explanation: <text>

    Args:
        completion: The model's critic response.
        prefix: Conversation messages [{content, role}]

    Returns:
        Float reward in range [0, 1].
    """

    reward = 1.0

    if completion != completion.lower():
        print("Completion not all lowercase; setting reward to 0.0")
        return 0.0

    if not re.search(
        r"^\s*agreement:\s*(yes|no)\s*$", completion, re.IGNORECASE | re.MULTILINE
    ):
        print("Missing or invalid 'agreement' line; setting reward to 0.0")
        return 0.0
    if not re.search(
        r"^\s*explanation:\s*.+", completion, re.IGNORECASE | re.MULTILINE | re.DOTALL
    ):
        print("Missing 'explanation' line; setting reward to 0.0")
        return 0.0

    agreement_match = re.search(
        r"^\s*agreement:\s*(yes|no)\s*$", completion, re.IGNORECASE | re.MULTILINE
    )
    agreement = agreement_match.group(1).lower() if agreement_match else None

    feedback = None
    guess = None

    last_message = prefix[-1].get("content", "")
    feedback_match = re.search(
        r"guess_feedback:\s*([^\n]+)", last_message, re.IGNORECASE
    )
    if feedback_match:
        feedback = feedback_match.group(1).strip()
        print(f"Extracted feedback: {feedback}")

    guess_match = re.search(r"guess:\s*(\w+)", last_message, re.IGNORECASE)
    if guess_match:
        guess = guess_match.group(1).lower().strip()
        print(f"Extracted guess from prompt: {guess}")
        if not (guess and len(guess) == 5 and guess.isalpha()):
            print("Invalid guess; setting reward to 0.0")
            return 0.0
    else:
        print("No guess found; setting reward to 0.0")
        return 0.0

    if feedback and guess:
        adherence_score = calculate_feedback_adherence(guess, feedback)
        print(f"Feedback adherence score (0..0.30): {adherence_score}")

        expected_agree = adherence_score >= 0.18
        model_agrees = agreement == "yes"

        if (expected_agree and model_agrees) or (
            not expected_agree and not model_agrees
        ):
            print("Agreement matches heuristic")
            return adherence_score / 0.3
        else:
            print("Agreement contradicts heuristic; setting reward to 0.0")
            return 0.0
    else:
        print("No usable feedback/guess; leaving reward unchanged")
        return 0.0

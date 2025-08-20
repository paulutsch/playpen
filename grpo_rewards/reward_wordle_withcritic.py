import logging
import re

from .reward_wordle import calculate_feedback_adherence

logger = logging.getLogger(__name__)


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

    if completion != completion.lower():
        logger.debug("Completion not all lowercase; returning 0.0")
        return 0.0

    if not re.search(
        r"^\s*agreement:\s*(yes|no)\s*$", completion, re.IGNORECASE | re.MULTILINE
    ):
        logger.debug("Missing or invalid 'agreement' line; returning 0.0")
        return 0.0
    if not re.search(
        r"^\s*explanation:\s*.+", completion, re.IGNORECASE | re.MULTILINE | re.DOTALL
    ):
        logger.debug("Missing 'explanation' line; returning 0.0")
        return 0.0

    agreement_match = re.search(
        r"^\s*agreement:\s*(yes|no)\s*$", completion, re.IGNORECASE | re.MULTILINE
    )
    agreement = agreement_match.group(1).lower() if agreement_match else None

    reward = 0.5
    logger.debug(f"Format compliance passed, base reward: {reward}")

    explanation_match = re.search(
        r"^\s*explanation:\s*(.+)", completion, re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    if explanation_match and explanation_match.group(1).strip():
        reward += 0.1
        logger.debug(f"Explanation present; reward now: {reward}")

    feedback = None
    guess = None
    if prefix:
        last_message = prefix[-1].get("content", "")
        feedback_match = re.search(
            r"guess_feedback:\s*([^\n]+)", last_message, re.IGNORECASE
        )
        if feedback_match:
            feedback = feedback_match.group(1).strip()
            logger.debug(f"Extracted feedback: {feedback}")

        guess_match = re.search(r"guess:\s*(\w+)", last_message, re.IGNORECASE)
        if guess_match:
            guess = guess_match.group(1).lower().strip()
            logger.debug(f"Extracted guess from prompt: {guess}")

    if feedback and guess and len(guess) == 5 and guess.isalpha():
        adherence_score = calculate_feedback_adherence(guess, feedback)
        logger.debug(f"Feedback adherence score (0..0.30): {adherence_score}")

        expected_agree = adherence_score >= 0.18
        model_agrees = agreement == "yes"

        if (expected_agree and model_agrees) or (
            not expected_agree and not model_agrees
        ):
            reward += 0.4
            logger.debug(f"Agreement matches heuristic; reward now: {reward}")
        else:
            logger.debug("Agreement contradicts heuristic; no bonus added")
    else:
        reward += 0.4
        logger.debug(
            f"No usable feedback/guess; neutral credit added, reward now: {reward}"
        )

    reward = max(0.0, min(1.0, reward))
    logger.debug(f"Final reward: {reward}")
    return reward

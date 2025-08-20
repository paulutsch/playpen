import logging

logger = logging.getLogger(__name__)


def reward_referencegame(completion: str, prefix: list[dict[str, str]]) -> float:
    logger.debug(f"reward_referencegame called with completion: {completion[:100]}...")
    logger.debug(f"prefix length: {len(prefix)}")
    logger.debug("Placeholder function - returning 0.0")
    return 0.0

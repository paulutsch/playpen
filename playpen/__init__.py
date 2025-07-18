import os

os.environ["CLEM_DISABLE_BANNER"] = "1"

BANNER = \
    r"""
.--------------..--------------..--------------..--------------..--------------..--------------..--------------.
|   ______     ||   _____      ||      __      ||  ____  ____  ||   ______     ||  _________   || ____  _____  |
|  |_   __ \   ||  |_   _|     ||     /  \     || |_  _||_  _| ||  |_   __ \   || |_   ___  |  |||_   \|_   _| |
|    | |__) |  ||    | |       ||    / /\ \    ||   \ \  / /   ||    | |__) |  ||   | |_  \_|  ||  |   \ | |   |
|    |  ___/   ||    | |   _   ||   / ____ \   ||    \ \/ /    ||    |  ___/   ||   |  _|  _   ||  | |\ \| |   |
|   _| |_      ||   _| |__/ |  || _/ /    \ \_ ||    _|  |_    ||   _| |_      ||  _| |___/ |  || _| |_\   |_  |
|  |_____|     ||  |________|  |||____|  |____|||   |______|   ||  |_____|     || |_________|  |||_____|\____| |
'--------------''--------------''--------------''--------------''--------------''--------------''--------------'
"""  # Blocks font, thanks to http://patorjk.com/software/taag/

if os.getenv("PLAYPEN_DISABLE_BANNER", "0") not in ("1", "true", "yes", "on"):
    print(BANNER)

from contextlib import contextmanager
from typing import List, Callable

from playpen.buffers import RolloutBuffer, BranchingRolloutBuffer, StepRolloutBuffer
from playpen.callbacks import BaseCallback, GameRecordCallback, RolloutProgressCallback, CallbackList
from playpen.base import BasePlayPen
from playpen.envs import PlayPenEnv
from playpen.envs.game_env import GameEnv
from playpen.envs.branching_env import GameBranchingEnv

__all__ = [
    "BaseCallback",
    "GameRecordCallback",
    "RolloutProgressCallback",
    "CallbackList",
    "BasePlayPen",
    "PlayPenEnv",
    "RolloutBuffer",
    "BranchingRolloutBuffer",
    "StepRolloutBuffer",
    "GameEnv",
    "GameBranchingEnv",
    "make_tree_env",
    "make_env"
]

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, benchmark


@contextmanager
def make_env(game_spec: GameSpec,
             players: List[Model],
             *,
             instances_filename: str = None,
             shuffle_instances: bool = False,
             task_iterations: int = None
             ) -> GameEnv:
    """
    Create an env that allows to collect conversations by letting the player model's
    play the game instances as specified by the games' instance file.

    :param game_spec: The game spec of the game to be loaded.
    :param players: The models to play the game. Order is important for role assignment.
        See the "roles" attribute in the game spec.
    :param instances_filename: The name of an instances file in the game's directory.
        The files specify the game instances to be played during the rollout.
    :param shuffle_instances: Whether to shuffle the instance before playing or not.
    :param task_iterations: A number that specified how often all instances should be played.
        This is useful, when all instances should be played exactly N times, like epochs.
        Default: Unlimited iterations.
    :return: The GameEnv
    """
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_filename=instances_filename) as game:
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameEnv(game, players,
                      task_iterator=task_iterator,
                      task_iterations=task_iterations)


@contextmanager
def make_tree_env(game_spec: GameSpec,
                  players: List[Model],
                  *,
                  instances_filename: str = None,
                  shuffle_instances: bool = False,
                  task_iterations: int = None,
                  branching_factor: int = 2,
                  branching_criteria: Callable[[GameEnv], bool] = None
                  ) -> GameBranchingEnv:
    """
    Create an env that evolves in a tree-like structure, that is,
    the env allows to create at certain steps of the conversation
    independently ongoing branches. This allows to collect at each
    step multiple responses for the same context.

    :param game_spec: The game spec of the game to be loaded.
    :param players: The models to play the game. Order is important for role assignment.
        See the "roles" attribute in the game spec.
    :param instances_filename: The name of an instances file in the game's directory.
        The files specify the game instances to be played during the rollout.
    :param shuffle_instances: Whether to shuffle the instance before playing or not.
    :param task_iterations: A number that specified how often all instances should be played.
        This is useful, when all instances should be played exactly N times, like epochs.
        Default: Unlimited iterations.
    :param branching_factor: The number of branches at each step when the branching_criteria is fulfilled.
    :param branching_criteria: A criteria to determine, when to branch at a step. Default: At every step.
    :return: The GameBranchingEnv
    """
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_filename=instances_filename) as game:
        assert branching_factor > 1, "The branching factor must be greater than one, otherwise use make_env"
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameBranchingEnv(game, players,
                               task_iterator=task_iterator,
                               task_iterations=task_iterations,
                               branching_factor=branching_factor,
                               branching_criteria=branching_criteria)

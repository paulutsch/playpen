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
from typing import List

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
def make_env(game_spec: GameSpec, players: List[Model],
             instances_filename: str = None, shuffle_instances: bool = False, task_iterations=None) -> GameEnv:
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_filename=instances_filename) as game:
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameEnv(game, players, task_iterator, task_iterations=task_iterations)


@contextmanager
def make_tree_env(game_spec: GameSpec, players: List[Model],
                  instances_filename: str = None, shuffle_instances: bool = False, task_iterations: int = None,
                  branching_factor: int = 2, branching_criteria=None) -> GameBranchingEnv:
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_filename=instances_filename) as game:
        assert branching_factor > 1, "The branching factor must be greater than one, otherwise use make_env"
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameBranchingEnv(game, players, task_iterator, task_iterations,
                               branching_factor=branching_factor,
                               branching_criteria=branching_criteria)

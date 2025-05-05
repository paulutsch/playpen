from contextlib import contextmanager
from typing import List

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, benchmark

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


@contextmanager
def make_env(game_spec: GameSpec, players: List[Model],
             instances_name: str = None, shuffle_instances: bool = False):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_name=instances_name) as game:
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameEnv(game, players, task_iterator)


@contextmanager
def make_tree_env(game_spec: GameSpec, players: List[Model],
                  instances_name: str = None, shuffle_instances: bool = False,
                  branching_factor: int = 2, branching_model=None):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_name=instances_name) as game:
        assert branching_factor > 1, "The branching factor must be greater than one"
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameBranchingEnv(game, players, task_iterator,
                               branching_factor=branching_factor, branching_model=branching_model)

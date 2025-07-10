import os
from copy import deepcopy
from typing import List, Tuple, Dict, Callable, Union

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, DialogueGameMaster, GameInstanceIterator, DefaultGameRecorder, Player
from clemcore.clemgame.resources import store_file
from clemcore.clemgame.benchmark import to_model_results_folder, to_player_model_infos

from playpen.envs import PlayPenEnv


class GameEnv(PlayPenEnv):

    def __init__(self, game: GameBenchmark, player_models: List[Model],
                 task_iterator: GameInstanceIterator, task_iterations: int = None, initial_reset=True):
        super().__init__()
        self._game = game
        self._game_name = game.game_name
        self._player_models = player_models
        self._model_results_folder = to_model_results_folder(player_models)
        self._task_iterator = task_iterator
        if len(self._task_iterator) < 1:
            raise RuntimeError(f"No game instances given for the game: '{self._game_name}'")
        # variables initialized on reset()
        self._game_instance: Dict = None
        self._experiment: Dict = None
        self._master: DialogueGameMaster = None
        self._task_iterations = task_iterations
        self._current_task_iteration = 0
        if initial_reset:  # if reset, then the game env is fully functional after init
            self.reset()

    def __deepcopy__(self, memo):
        _copy = type(self).__new__(self.__class__)
        memo[id(self)] = _copy
        _copy.__dict__.update(self.__dict__.copy())  # shallow copy of most attributes (for now)
        _copy._master = deepcopy(self._master)
        _copy._task_iterator = deepcopy(self._task_iterator)
        return _copy

    @property
    def initial_prompts(self):
        return [{player: player.initial_prompt} for player in self.master.get_players()]

    @property
    def experiment(self):
        return self._experiment

    @property
    def master(self):
        return self._master

    @master.setter
    def master(self, master):
        self._master = master

    def reset(self) -> None:
        try:
            self._experiment, self._game_instance = next(self._task_iterator)
            self.master = self._game.create_game_master(self._experiment, self._player_models)
            self.master.game_recorder = DefaultGameRecorder(self._game_name,
                                                            self._experiment["name"],
                                                            self._game_instance["game_id"],
                                                            self._model_results_folder,
                                                            to_player_model_infos(self._player_models))
            self.master.setup(**self._game_instance)
        except StopIteration as e:
            self._current_task_iteration += 1
            if self._task_iterations is not None and self._current_task_iteration >= self._task_iterations:
                raise e
            self._task_iterator.reset()
            self.reset()

    def observe(self) -> Tuple[Player | Callable, Dict | List[Dict]]:
        player = self.master.current_player
        context = self.master.get_context_for(player)
        return player, context

    def step(self, response: str | List) -> Tuple[bool | List, Dict | List]:
        self._done, info = self.master.step(response)
        return self._done, info

    def store_records(self, top_dir: str, rollout_dir: str, episode_dir: str):
        experiment_dir = f"{self.experiment['index']}_{self.experiment['name']}"
        experiment_path = os.path.join(top_dir,
                                       self._model_results_folder,
                                       rollout_dir,
                                       self._game_name,
                                       experiment_dir)
        episode_path = os.path.join(experiment_path, episode_dir)
        store_file(self.experiment, f"experiment.json", experiment_path)
        store_file(self._game_instance, f"instance.json", episode_path)
        store_file(self.master.game_recorder.interactions, f"interactions.json", episode_path)
        store_file(self.master.game_recorder.requests, f"requests.json", episode_path)

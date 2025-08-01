from typing import List, Dict
from datasets import Dataset

import clemcore.backends as cb
from clemcore.clemgame import Player
from clemcore.backends import Model

from playpen.envs.game_env import GameEnv
from playpen.envs.branching_env import GameBranchingEnv, GameTree, ResponseTreeNode


class RolloutBuffer:

    def __init__(self, game_env):
        self.game_env = game_env
        self.initial_prompts: Dict[Player, Dict] = {}  # initial prompts that are not given in the initial context

    def on_step(self, context, response, done, info):
        pass

    def on_done(self):
        pass

    def reset(self):
        pass

    def to_conversational_dataset(self, perspective: cb.Model) -> Dataset:
        """
        Converts the data collected in the buffer into a dataset where each row represents a conversation.

        A conversation is basically a dict where the "messages" entry points to a list of dicts where each
        of these contain alternating entries of "role" (assistant or user) and "content" depending on the perspective.
        For example:

        {"messages" = [
            {"role": "user", "content": "Hello, how are you?"},\n
            {"role": "assistant", "content": "I'm doing great. How can I help you today?"},\n
            {"role": "user", "content": "I'd like to show off how chat templating works!"}\n
            ]}

        See also https://huggingface.co/docs/trl/dataset_formats#conversational
        Args:
            perspective: to take in the dataset as specified by the given model
        Returns: the Dataset
        """
        pass


class StepRolloutBuffer(RolloutBuffer):
    """ This buffer can collect the trajectories generated by a game env """

    def __init__(self, game_env: GameEnv):
        assert isinstance(game_env, GameEnv), "StepRolloutBuffer can only be used with GameEnv"
        super().__init__(game_env)
        self.trajectories: List = None
        self.current_trajectory: int = None
        self.reset()

    def on_step(self, context, response, done, info):
        step = dict(context=context, response=response, done=done, info=info)
        self.trajectories[self.current_trajectory].append(step)

    def on_done(self):
        self.trajectories.append([])
        self.current_trajectory += 1

    def reset(self):
        self.trajectories = [[]]
        self.current_trajectory = 0


class BranchingRolloutBuffer(RolloutBuffer):
    """ This buffer can collect the trajectories generated by the branching env """

    def __init__(self, game_env: GameBranchingEnv):
        assert isinstance(game_env, GameBranchingEnv), "TreeRolloutBuffer can only be used with GameBranchingEnv"
        super().__init__(game_env)
        self.forest: List[GameTree] = None
        self.reset()

    def on_done(self):
        active_tree = self.game_env.get_active_tree()
        self.forest.append(active_tree)

    def reset(self):
        self.forest = []

    def to_conversational_dataset(self, perspective: Model) -> Dataset:
        def recursive_add_to(_messages: List[Dict], node: ResponseTreeNode):
            # only collect for given conversational perspective
            player_model = node.unwrap().master.current_player.model
            if perspective is player_model:
                # we reverse later
                _messages.append(dict(role="assistant", content=node.response))
                _messages.append(node.context)
            if isinstance(node.parent, ResponseTreeNode):
                recursive_add_to(_messages, node.parent)
            return _messages

        dataset = []
        for active_tree in self.forest:
            for leave in active_tree.find_leaves():
                messages = recursive_add_to([], leave)
                messages.reverse()
                dataset.append(dict(messages=messages, reward=leave.info["episode_score"]))
        return Dataset.from_list(dataset)

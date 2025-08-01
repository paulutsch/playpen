import os
from copy import deepcopy, copy
from typing import List, Dict, Callable, Tuple, Union, Set

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameInstanceIterator, Player

from playpen.envs import PlayPenEnv
from playpen.envs.game_env import GameEnv


class BranchingResponse:

    def __init__(self, parent_env: GameEnv, branch_env: GameEnv, parent_context: Dict, branch_response: str):
        self.parent_env = parent_env
        self.branch_env = branch_env
        self.parent_context = parent_context
        self.branch_response = branch_response

    def step(self):
        return self.branch_env.step(self.branch_response)

    def __str__(self):
        return self.branch_response


class BranchingCandidate:

    def __init__(self, response: BranchingResponse, done: bool, info: Dict):
        self.response = response
        self.done = done
        self.info = info

    def add_branch_to(self, game_tree):
        """ Find parent node and add child"""
        parent_node = game_tree.find_node(self.response.parent_env)
        assert parent_node is not None, "There must be a parent node that wraps the candidates parent env"
        branch_node = ResponseTreeNode(self.response.branch_env,
                                       self.response.parent_context,
                                       self.response.branch_response,
                                       self.done, self.info)
        parent_node.connect_to(branch_node)


class GameBranchingEnv(PlayPenEnv):
    """
    Create an env that evolves in a tree-like structure, that is,
    the env allows to create at certain steps of the conversation
    independently ongoing branches. This allows to collect at each
    step multiple responses for the same context.

    :param game: The game to be played
    :param player_models: The models to play the game. Order is important for role assignment.
        See the "roles" attribute in the game spec.
    :param task_iterator: The iterator that returns the sequence of instances to be played.
    :param task_iterations: A number that specified how often all instances should be played.
        This is useful, when all instances should be played exactly N times, like epochs.
        Default: Unlimited iterations.
    :param branching_factor: The number of branches at each step when the branching_criteria is fulfilled.
    :param branching_criteria: A criteria to determine, when to branch at a step. Default: At every step.
    :return: The GameBranchingEnv
    """

    def __init__(self,
                 game: GameBenchmark,
                 player_models: List[Model],
                 *,
                 task_iterator: GameInstanceIterator,
                 task_iterations: int = None,
                 branching_factor: int = 2,
                 branching_criteria: Callable[[GameEnv], bool] = None):
        super().__init__()
        assert branching_factor > 0, "The branching factor must be greater than zero"
        self._root: GameEnv = GameEnv(game, player_models, task_iterator=task_iterator, task_iterations=task_iterations)
        self._game_tree = GameTree(GameTreeNode(self._root))
        self._active_envs: List[GameEnv] = [self._root]
        self._branching_factor: int = branching_factor
        self._branching_criteria = branching_criteria

    @property
    def initial_prompts(self):
        return [{player: player.initial_prompt} for player in self._root.master.get_players()]

    def reset(self) -> None:  # all game branches always operate on the same task / episode
        self._root.reset()
        self._game_tree = GameTree(GameTreeNode(self._root))
        self._active_envs: List[GameEnv] = [self._root]

    def observe(self) -> Tuple[Union[Player, Callable], Union[Dict, List[Dict]]]:
        contexts: List[Dict] = []
        players: List[Player] = []
        for game_env in self._active_envs:
            player, context = game_env.observe()
            players.append(player)
            contexts.append(context)
        # GameBranchingPlayer assumes that (parent_env, parent_context) can be re-assembled by zipping (using the order)
        branching_player = GameBranchingPlayer(self._active_envs, players,
                                               self._branching_factor, self._branching_criteria)
        return branching_player, contexts

    def step(self, responses: Union[str, List]) -> Tuple[Union[bool, List], Union[Dict, List]]:
        assert isinstance(responses, list), f"GameTreeEnv expects a list of responses and not {responses.__class__}"

        context_dones = []
        context_infos = []
        candidates: List[BranchingCandidate] = []  # called candidates because we considered to apply a pruning function
        for context_responses in responses:
            response_dones = []
            response_infos = []
            for response in context_responses:  # each response represents a possible branch in the tree
                done, info = response.step()
                response_dones.append(done)
                response_infos.append(info)
                candidate = BranchingCandidate(response, done, info)
                candidates.append(candidate)
            context_dones.append(response_dones)
            context_infos.append(response_infos)

        self._done = all([candidate.done for candidate in candidates])

        self._active_envs = []  # memorize active leaves so that we do not have to find them again
        for candidate in candidates:
            candidate.add_branch_to(self._game_tree)
            self._active_envs.append(candidate.response.branch_env)

        # return all dones and infos so that they match the quantity of the responses
        return context_dones, context_infos

    def store_records(self, top_dir: str, rollout_dir: str, episode_dir: str):
        for branch_idx, game_env in enumerate(self._active_envs):
            game_env.store_records(top_dir, rollout_dir, os.path.join(episode_dir, f"branch_{branch_idx}"))

    def get_active_tree(self) -> "GameTree":
        """ Ad-hoc calculation of the tree containing only active branches """
        leaves = self._game_tree.find_leaves()
        active_leaves = []
        for leave in leaves:
            if leave.unwrap() in self._active_envs:
                active_leaves.append(leave)

        def label_active_recursive(active_node):
            active_node.tag("active")
            if active_node.parent:  # root has no parent
                label_active_recursive(active_node.parent)

        for active_leave in active_leaves:
            label_active_recursive(active_leave)

        def copy_active_tree_recursive(active_node):
            _copy = copy(active_node)
            _copy.branches = [node for node in active_node if node.has_tag("active")]
            for branch in _copy:
                copy_active_tree_recursive(branch)

        active_root = copy(self._game_tree.root)  # we do not want to change the initial tree
        copy_active_tree_recursive(active_root)
        return GameTree(active_root)

    def is_done(self) -> bool:
        return self._done


class GameBranchingPlayer(Callable):
    """    Applies a player to a given context as many times as determined by the branching factor. """

    def __init__(self, current_envs: List[GameEnv], current_players: List[Player],
                 branching_factor: int = 1, branching_criteria: Callable[[GameEnv], bool] = None):
        assert branching_factor > 0, "The branching factor must be greater than zero"
        self._branching_factor = branching_factor
        self._do_branch = branching_criteria or (lambda parent_env: True)  # always
        self._current_envs = current_envs
        self._current_players = current_players

    def __call__(self, contexts: List[str]) -> List[List[BranchingResponse]]:
        """
        For each context we return multiple responses that possibly transition the environment.
        :param contexts:
        :return:
        """
        assert isinstance(contexts, List), "The context for TreePlayer must be a list of game environments"
        assert len(self._current_envs) == len(contexts), "There must be as many active branches as given contexts"
        context_responses = []
        for parent_env, parent_context in zip(self._current_envs, contexts):
            branch_responses = []
            branching_factor = self._branching_factor if self._do_branch(parent_env) else 1
            for _ in range(branching_factor):
                # We need to copy the env even with factor=1 (for the teacher) b.c. otherwise we run into problems
                # when adding the response to the tree, since we use the env identity as an id. If we do not copy,
                # then there will be two nodes with the same env which makes finding them via the env unpredictable.
                branch_env: GameEnv = deepcopy(parent_env)
                branch_player = branch_env.master.current_player  # we use the branch player as it keeps state
                # this already changes the player state in branch env
                branch_response = branch_player(parent_context)
                branch_responses.append(BranchingResponse(parent_env, branch_env, parent_context, branch_response))
            context_responses.append(branch_responses)
        return context_responses


class GameTreeNode:
    def __init__(self, game_env: GameEnv):
        self._game_env = game_env
        self._branches: List[GameTreeNode] = []
        self._parent: GameTreeNode = None  # root has no parent
        self._tags: Set = set()

    @property
    def branches(self):
        return self._branches

    @branches.setter
    def branches(self, branches):
        self._branches = branches

    @property
    def parent(self):
        return self._parent

    @property
    def tags(self):
        return self._tags

    def untag(self, tag: str):
        self._tags.remove(tag)

    def tag(self, tag: str):
        self._tags.add(tag)

    def has_tag(self, tag: str):
        return tag in self._tags

    def __iter__(self):
        return iter(self._branches)

    def __bool__(self):
        return bool(self._branches)

    def unwrap(self):
        return self._game_env

    def wraps(self, game_env: GameEnv) -> bool:
        is_wrapping = self._game_env is game_env
        return is_wrapping

    def connect_to(self, branch_node: "GameTreeNode"):
        if branch_node in self._branches:
            return
        self._branches.append(branch_node)
        branch_node._parent = self


class ResponseTreeNode(GameTreeNode):

    def __init__(self, game_env: GameEnv, context: Dict, response: str, done: bool, info: Dict):
        super().__init__(game_env)
        self.context = context
        self.response = response
        self.done = done
        self.info = info


class GameTree:

    def __init__(self, root: GameTreeNode):
        self._root: GameTreeNode = root

    @property
    def root(self):
        return self._root

    def find_node(self, target_env: GameEnv):
        def _find_node(node):
            if node.wraps(target_env):  # check for object identity
                return node
            for branch in node:
                target_node = _find_node(branch)
                if target_node is not None:
                    return target_node
            return None

        return _find_node(self._root)

    def find_leaves(self):
        def _find_leaves(node):
            if not node:
                return [node]
            leaves = []
            for branch in node:
                leaves.extend(_find_leaves(branch))
            return leaves

        return _find_leaves(self._root)

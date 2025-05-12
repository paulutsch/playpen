import argparse
import inspect
import importlib.util as importlib_util
import os

import clemcore.cli as clem
from clemcore.backends import ModelSpec, ModelRegistry, BackendRegistry
from clemcore.clemgame import GameRegistry
from playpen import BasePlayPen


def train(file_path: str, learner: ModelSpec, teacher: ModelSpec, temperature: float, max_tokens: int):
    def is_playpen(obj):
        return inspect.isclass(obj) and issubclass(obj, BasePlayPen) and obj is not BasePlayPen

    try:
        file_name = os.path.splitext(file_path)[0]
        spec = importlib_util.spec_from_file_location(file_name, file_path)
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        playpen_subclasses = inspect.getmembers(module, predicate=is_playpen)
        if len(playpen_subclasses) == 0:
            raise ValueError(f"Cannot load playpen trainer, because no BasePlayPen found in {file_path}.\n"
                             f"Make sure that you have implemented a subclass of BasePlayPen and try again.")
        _, playpen_cls = playpen_subclasses[0]
    except Exception as e:
        raise RuntimeError(f"Cannot load playpen trainer, because {e}")

    game_registry = GameRegistry.from_directories_and_cwd_files()
    model_registry = ModelRegistry.from_packaged_and_cwd_files()

    learner_spec = model_registry.get_first_model_spec_that_unify_with(learner)
    print(f"Found registered model spec that unifies with {learner.to_string()} -> {learner_spec}")

    teacher_spec = model_registry.get_first_model_spec_that_unify_with(learner)
    print(f"Found registered model spec that unifies with {teacher.to_string()} -> {teacher_spec}")

    backend_registry = BackendRegistry.from_packaged_and_cwd_files()
    for model_spec in [learner_spec, teacher_spec]:
        backend_selector = model_spec.backend
        if not backend_registry.is_supported(backend_selector):
            raise ValueError(f"Specified model backend '{backend_selector}' not found in backend registry.")
        print(f"Found registry entry for backend {backend_selector} "
              f"-> {backend_registry.get_first_file_matching(backend_selector)}")

    print(f"Dynamically import backend {learner_spec.backend}")
    backend = backend_registry.get_backend_for(learner_spec.backend)
    learner_model = backend.get_model_for(learner_spec)
    learner_model.set_gen_args(max_tokens=max_tokens, temperature=temperature)
    print(f"Successfully loaded {learner_spec.model_name} model")

    print(f"Dynamically import backend {teacher_spec.backend}")
    backend = backend_registry.get_backend_for(teacher_spec.backend)
    teacher_model = backend.get_model_for(teacher_spec)
    teacher_model.set_gen_args(max_tokens=max_tokens, temperature=temperature)
    print(f"Successfully loaded {teacher_spec.model_name} model")

    playpen_cls(learner_model, teacher_model).learn(game_registry)


def cli(args: argparse.Namespace):
    if args.command_name == "list":
        if args.mode == "games":
            clem.list_games(args.selector, args.verbose)
        elif args.mode == "models":
            clem.list_models(args.verbose)
        elif args.mode == "backends":
            clem.list_backends(args.verbose)
        else:
            print(f"Cannot list {args.mode}. Choose an option documented at 'list -h'.")
    if args.command_name == "run":
        train(args.file_path, ModelSpec.from_string(args.learner), ModelSpec.from_string(args.teacher),
              args.temperature, args.max_tokens)


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    list_parser = sub_parsers.add_parser("list")
    list_parser.add_argument("mode", choices=["games", "models", "backends"],
                             default="games", nargs="?", type=str,
                             help="Choose to list available games, models or backends. Default: games")
    list_parser.add_argument("-v", "--verbose", action="store_true")
    list_parser.add_argument("-s", "--selector", type=str, default="all")

    train_parser = sub_parsers.add_parser("run")
    train_parser.add_argument("file_path", type=str,
                              help="The path to the trainer file to use for learning.")
    train_parser.add_argument("-l", "--learner", type=str,
                              help="The model name of the learner model (as listed by 'playpen list models').")
    train_parser.add_argument("-t", "--teacher", type=str,
                              help="The model name of the partner model (as listed by 'playpen list models').")
    train_parser.add_argument("-T", "--temperature", type=float, required=False, default=0.0)
    train_parser.add_argument("-L", "--max_tokens", type=int, required=False, default=300)

    cli(parser.parse_args())


if __name__ == "__main__":
    main()

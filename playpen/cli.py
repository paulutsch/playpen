import argparse
import inspect
import os

import clemcore.cli as clem
from clemcore.backends import ModelSpec, ModelRegistry, BackendRegistry
from clemcore.clemgame import GameRegistry
from clemcore.playpen import BasePlayPen


def train(learner: ModelSpec, teacher: ModelSpec, lookup_name: str):
    import importlib.util as importlib_util

    def is_playpen(obj):
        return inspect.isclass(obj) and issubclass(obj, BasePlayPen) and obj is not BasePlayPen

    lookup_name = lookup_name if lookup_name.endswith(".py") else lookup_name + ".py"
    playpen_cls = None
    for file_name in os.listdir():
        if file_name == lookup_name:
            module_path = os.path.join(os.getcwd(), file_name)
            spec = importlib_util.spec_from_file_location(os.path.splitext(lookup_name)[0], module_path)
            module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(module)
            playpen_subclasses = inspect.getmembers(module, predicate=is_playpen)
            _, playpen_cls = playpen_subclasses[0]
            break

    if playpen_cls is None:
        raise RuntimeError(f"No playpen trainer found in file '{lookup_name}'")

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
    learner_model.set_gen_args(max_tokens=300, temperature=0.0)
    print(f"Successfully loaded {learner_spec.model_name} model")

    print(f"Dynamically import backend {teacher_spec.backend}")
    backend = backend_registry.get_backend_for(teacher_spec.backend)
    teacher_model = backend.get_model_for(teacher_spec)
    teacher_model.set_gen_args(max_tokens=300, temperature=0.0)
    print(f"Successfully loaded {teacher_spec.model_name} model")

    playpen_cls(learner_model, teacher_model).learn_interactive(game_registry)


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
    if args.command_name == "train":
        train(ModelSpec.from_string(args.learner), ModelSpec.from_string(args.teacher), args.lookup_name)


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    list_parser = sub_parsers.add_parser("list")
    list_parser.add_argument("mode", choices=["games", "models", "backends"],
                             default="games", nargs="?", type=str,
                             help="Choose to list available games, models or backends. Default: games")
    list_parser.add_argument("-v", "--verbose", action="store_true")
    list_parser.add_argument("-s", "--selector", type=str, default="all")

    train_parser = sub_parsers.add_parser("train")
    train_parser.add_argument("-l", "--learner", type=str)
    train_parser.add_argument("-t", "--teacher", type=str)
    train_parser.add_argument("-f", "--lookup_name", type=str, default="trainer", required=False,
                              help="The name of the trainer file to use for learning (without .py extension)")

    cli(parser.parse_args())


if __name__ == "__main__":
    main()

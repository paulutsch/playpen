from .clembench_grpo_rewards import get_reward_fn_by_game

def _messages_to_text(p):
    if isinstance(p, list):
        return "\n".join(m.get("content", "") for m in p if isinstance(m, dict))
    if isinstance(p, dict) and "content" in p:
        return p["content"]
    return str(p)

def _detect_game_from_prompt_text(txt: str) -> str:
    s = txt.lower()
    if "welcome to wordle" in s and "critic" in s:
        return "wordle_withcritic"
    if "welcome to wordle" in s and "clue:" in s:
        return "wordle_withclue"
    if "welcome to wordle" in s:
        return "wordle"
    if "you are playing a collaborative word guessing game" in s or "related words are" in s:
        return "taboo"
    if "target grid" in s and "distractor grid" in s:
        return "referencegame"
    if "what is your next instruction?" in s:
        return "imagegame"
    return "wordle"

def router_reward(completions, **kwargs):
    prompts = kwargs.get("prompts") or kwargs.get("prompt") or []
    games = kwargs.get("game")
    if games is None:
        games = []
        for p in prompts:
            games.append(_detect_game_from_prompt_text(_messages_to_text(p)))

    def per_i_kwargs(i: int):
        out = {}
        for k, v in kwargs.items():
            if k in {"completions", "prompts", "prompt"}:
                continue
            if isinstance(v, (list, tuple)) and len(v) > i:
                out[k] = v[i]
            else:
                out[k] = v
        if prompts:
            out.setdefault("prompts", [prompts[i]])
        if isinstance(games, (list, tuple)) and len(games) > i:
            out["game"] = games[i]
        else:
            out["game"] = games
        return out

    scores = []
    for i, comp in enumerate(completions):
        game_i = games[i] if isinstance(games, (list, tuple)) and len(games) > i else games
        fn = get_reward_fn_by_game(game_i)
        if fn is None:
            fn = get_reward_fn_by_game("wordle")
        scores.append(fn([comp], **per_i_kwargs(i))[0])
    return scores


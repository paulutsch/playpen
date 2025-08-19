from grpo_rewards.clembench_grpo_rewards import (
    reward_wordle, reward_wordle_withcritic, reward_wordle_withclue, reward_taboo, reward_referencegame, reward_imagegame
)

def test_wordle_withclue():
    completions = [
        [{"content": "guess: crane\nexplanation: The clue suggests ownership by others."}]
    ]
    scores = reward_wordle_withclue(
        completions,
        target="crane",
        clue="others'"
    )
    print("wordle_withclue scores:", scores)

def test_taboo():
    completions = [
        [{"content": "CLUE: opposite of yes"}]
    ]
    prompt_text = (
        "This is the target word that you need to describe...\n\n"
        "none\n\nRelated words are:\n- no\n- never\n- nobody\n\nImportant: ..."
    )
    scores = reward_taboo(completions, prompts=[prompt_text])
    print("taboo scores:", scores)

def test_referencegame():
    expr = [{"content": "Expression: the third row is fully X"}]
    prompt = (
        "Target grid:\n\n"
        "▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\nX X X X X\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n\n"
        "Distractor grid 1:\n\n"
        "X X X X X\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n\n"
        "Distractor grid 2:\n\n"
        "▢ ▢ ▢ ▢ ▢\nX X X X X\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n"
    )
    print("referencegame scores:", reward_referencegame([expr], prompts=[prompt]))

def test_imagegame():
    # Einfache Instruktion + DONE (korrekt oder nicht)
    completions = [
        [{"content": "Instruction: Fill the fifth row with X\nInstruction: DONE"}]
    ]
    target_grid = (
        "▢ ▢ ▢ ▢ S\n"
        "S S S S S\n"
        "▢ ▢ ▢ ▢ S\n"
        "▢ ▢ ▢ ▢ S\n"
        "S S S S S\n"
    )
    # Start-Grid leer; Ziel hat letzte Zeile mit S – unsere Instruktion macht X → Teilfortschritt, aber falscher Buchstabe
    print("imagegame scores:", reward_imagegame(completions, target_grid=target_grid))

def test_wordle_base_exact_match():
    # Exakter Treffer -> sollte nahe 1.0 sein
    completions = [[{"content": "guess: crane\nexplanation: starting with a strong candidate."}]]
    scores = reward_wordle(completions, target="crane")
    print("wordle (exact) scores:", scores)

def test_wordle_base_feedback_and_penalty():
    # Mit Feedback (kein Target): partieller Score
    completions_ok = [[{"content": "guess: apple\nexplanation: using provided feedback to refine."}]]
    fb = "a<yellow> p<yellow> p<green> l<yellow> e<red>"
    scores_ok = reward_wordle(completions_ok, feedback=fb)
    print("wordle (feedback) scores:", scores_ok)

    # Halluzinierte guess_feedback:-Zeile -> -0.2 Strafe
    completions_bad = [[{"content": "guess: apple\nexplanation: trying this.\nguess_feedback: a<green> p<green> p<green> l<green> e<green>"}]]
    scores_bad = reward_wordle(completions_bad)
    print("wordle (fabricated feedback penalty) scores:", scores_bad)

def test_wordle_withcritic_agree_no_change():
    # Kritiker "agree" + Erklärung erwähnt "agree" -> +0.15 Bonus
    completions = [[{"content": "guess: swing\nexplanation: I agree with the critic; keeping this choice based on the clue about others."}]]
    scores = reward_wordle_withcritic(
        completions,
        clue="others'",                 # damit die withclue-Komponente 0.2 holen kann (Erklärung enthält "others")
        critic_agreement="agree",
        prior_guess="swing"             # kein Wechsel -> kein +0.10
    )
    print("wordle_withcritic (agree, no change) scores:", scores)

def test_wordle_withcritic_disagree_change_and_target():
    # Kritiker "disagree" + Wechsel des Guess + Target bekannt -> sollte an 1.0 clampen
    completions = [[{"content": "guess: crane\nexplanation: I disagree with the critic; changing to the target aligned with others' meaning."}]]
    scores = reward_wordle_withcritic(
        completions,
        target="crane",                 # volle 0.6 Inhaltsbonus
        clue="others'",                 # +0.2, weil in der Erklärung erwähnt
        critic_agreement="disagree",    # +0.15, wenn 'disagree' erwähnt
        prior_guess="swing"             # Guess geändert -> +0.10
    )
    print("wordle_withcritic (disagree, change, target) scores:", scores)


if __name__ == "__main__":
    test_wordle_withclue()
    test_taboo()
    test_referencegame()
    test_imagegame()
    test_wordle_base_exact_match()                # <— NEU
    test_wordle_base_feedback_and_penalty()       # <— NEU
    test_wordle_withcritic_agree_no_change()      # <— NEU
    test_wordle_withcritic_disagree_change_and_target()  # <— NEU




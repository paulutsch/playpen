#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# clembench_grpo_rewards.py
# Reward functions for GRPO on clembench-style prompts/games.
# Compatible with TRL GRPOTrainer (main + v0.21 style).
#
# ->Each function returns one float per completion in range [-1,1].
# ->They try to parse needed context from kwargs["prompts"] (if provided) or from extra kwargs like target/clue/feedback. When not available, they fall back to format/compliance rewards.

#every function gives:
    #format rewards
    #content rewards
    #penalizing errors

import re
import math
from typing import List, Dict, Any

# ------------------------------ small utils ------------------------------

ORD2IDX = {
    "first": 0, "1st": 0, "one": 0, "1": 0, "top": 0, "leftmost": 0,
    "second": 1, "2nd": 1, "two": 1, "2": 1,
    "third": 2, "3rd": 2, "three": 2, "3": 2, "middle": 2, "center": 2, "centre": 2,
    "fourth": 3, "4th": 3, "four": 3, "4": 3,
    "fifth": 4, "5th": 4, "five": 4, "5": 4, "last": 4, "bottom": 4, "rightmost": 4,
}

#circumvent format issues (expected: list, dict, ...) transforming everything to text
def _as_text(sample):
    """Convert a TRL completion (string or list[{'content': str}]) to plain text."""
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict) and "content" in sample:
        return sample["content"]
    if isinstance(sample, list):
        # typical chat form: a list of messages; we take the assistant's content if present
        # but reward funcs normally receive only the generated assistant message as one-item list
        if sample and isinstance(sample[0], dict) and "content" in sample[0]:
            return sample[0]["content"]
        return "\n".join(_as_text(x) for x in sample)
    return str(sample)

#extracting prompts for parser: try to get needed information for rewards from prompts (can be addded individually by providing **kwargs to functions) 
#(only prompts from latest benchmark_version chosen=1.6v)
def _get_prompt_texts(kwargs: Dict[str, Any]) -> List[str]:
    """Try to pull original prompts per sample, tolerant to TRL variants."""
    prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("queries")
    if not prompts:
        return []
    out = []
    for p in prompts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, list):
            # conversational messages
            # pick the last user/developer/system chunk content as the prompt text surrogate
            parts = []
            for msg in p:
                if isinstance(msg, dict) and "content" in msg:
                    parts.append(msg["content"])
            out.append("\n".join(parts))
        elif isinstance(p, dict) and "content" in p:
            out.append(p["content"])
        else:
            out.append(str(p))
    return out

#normalization
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

#clip/constrain rewards
def _clip(x: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, x))

def _starts_with_tag(text: str, tag: str) -> bool:
    return _norm(text).startswith(tag.lower())

#search
def _contains(text: str, key: str) -> bool:
    return key.lower() in text.lower()

#tokenize
def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", s.lower())

#count useful words (for taboo: brief+substancial)
def _count_content_words(s: str) -> int:
    toks = _tokenize(s)
    stop = {"the","a","an","and","or","but","if","then","so","to","of","for","in","on","at","by","with","as","is","are","be"}
    return sum(1 for t in toks if t not in stop and len(t) > 2)

# ------------------------------ grid helpers ------------------------------

#converts ascii grid to lists
def _parse_grid_block(block: str) -> List[List[str]]:
    """
    Parse a 5x5 grid written with tokens like '▢' or letters ('X','S',...).
    Returns matrix[5][5] (strings).
    """
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    # accept compact "▢ ▢ X ..." style
    grid = []
    for ln in lines:
        cells = ln.split()
        # handle lines without spaces (unlikely here)
        if len(cells) == 1 and len(cells[0]) >= 5:
            cells = list(cells[0])
        grid.append(cells)
    # best effort to 5x5
    grid = [row[:5] + ["▢"]*(5-len(row)) for row in grid[:5]]
    if len(grid) < 5:
        grid += [["▢"]*5 for _ in range(5-len(grid))]
    return grid

#compute matching of cells
def _grid_equal(a: List[List[str]], b: List[List[str]]) -> bool:
    return all(a[r][c] == b[r][c] for r in range(5) for c in range(5))

#compute matching of cells
def _grid_match_count(a: List[List[str]], b: List[List[str]]) -> int:
    return sum(1 for r in range(5) for c in range(5) if a[r][c] == b[r][c])

def _empty_grid() -> List[List[str]]:
    return [["▢"]*5 for _ in range(5)]

#understands some simple instructions
def _apply_instruction(grid: List[List[str]], instr: str) -> None:
    """
    Apply a small subset of supported commands:
    - "Put an E in second row third column"
    - "Put a X in 5th row 1st column"
    - "Fill the last row with X"
    - "Fill the 2nd column with S"
    """
    s = _norm(instr)
    # Put an <L> in <row> row <col> column
    m = re.search(r"put (?:an|a)\s+([a-z])\s+in\s+(first|second|third|fourth|fifth|last|[1-5]|[0-9]+)\s+row\s+(first|second|third|fourth|fifth|last|[1-5]|[0-9]+)\s+column", s)
    if m:
        L = m.group(1).upper()
        r_tok, c_tok = m.group(2), m.group(3)
        r = ORD2IDX.get(r_tok, None)
        c = ORD2IDX.get(c_tok, None)
        if r is None and r_tok.isdigit(): r = max(0, min(4, int(r_tok)-1))
        if c is None and c_tok.isdigit(): c = max(0, min(4, int(c_tok)-1))
        if r is not None and c is not None:
            grid[r][c] = L
        return

    # Fill the <row/column> with <L>
    m = re.search(r"fill the\s+(first|second|third|fourth|fifth|last|[1-5])\s+(row|column)\s+with\s+([a-z])", s)
    if m:
        pos_tok, axis, L = m.group(1), m.group(2), m.group(3).upper()
        idx = ORD2IDX.get(pos_tok, None)
        if idx is None and pos_tok.isdigit():
            idx = max(0, min(4, int(pos_tok)-1))
        if idx is None: 
            return
        if axis == "row":
            for c in range(5): grid[idx][c] = L
        else:
            for r in range(5): grid[r][idx] = L
        return
    # otherwise: ignore (reward will not improve)

# ------------------------------ WORDLE core ------------------------------

def _extract_guess(text: str) -> str | None:
    m = re.search(r"(?im)^\s*guess:\s*([a-z]{5})\s*$", text)
    return m.group(1) if m else None

def _extract_explanation(text: str) -> str:
    m = re.search(r"(?is)explanation:\s*(.+)$", text)
    return m.group(1).strip() if m else ""

#penalizes self-invented feedback
def _has_fabricated_feedback(text: str) -> bool:
    return bool(re.search(r"(?i)guess_feedback\s*:", text))

#(green letters=1, yellow= 0.5, normalized by diving by 5)
def _wordle_score_from_target(guess: str, target: str) -> float:
    """Score in [0,1]: 1 for exact, else greens=1/5, yellows=0.5/5."""
    greens = sum(g == t for g, t in zip(guess, target))
    # yellows: count letters in both minus greens, capped by target letter counts
    from collections import Counter
    gC, tC = Counter(guess), Counter(target)
    common = sum(min(gC[ch], tC[ch]) for ch in gC)
    yellows = max(0, common - greens)
    return _clip((greens + 0.5*yellows) / 5.0, 0.0, 1.0)

def _parse_feedback_string(s: str) -> List[tuple[str,str]]:
    """Parse 'a<yellow> p<green> ...' -> [(a,'yellow'), (p,'green'), ...]"""
    pairs = []
    for m in re.finditer(r"([a-z])\s*<\s*(green|yellow|red)\s*>", s, re.I):
        pairs.append((m.group(1).lower(), m.group(2).lower()))
    return pairs

# ------------------------------ REFERENCEGAME helpers ------------------------------

def _rows_full_of_X(grid): return {r for r in range(5) if all(cell.upper()=="X" for cell in grid[r])}
def _cols_full_of_X(grid): return {c for c in range(5) if all(grid[r][c].upper()=="X" for r in range(5))}
def _main_diag_full_X(grid): return all(grid[i][i].upper()=="X" for i in range(5))
def _anti_diag_full_X(grid): return all(grid[i][4-i].upper()=="X" for i in range(5))

#extracts target/distractors from prompt
def _parse_three_grids_from_prompt(prompt_text: str):
    # Expect "Target grid:" ... "\n\nDistractor grid 1:" ... "\n\nDistractor grid 2:"
    parts = re.split(r"(?i)distractor grid\s*1\s*:\s*", prompt_text)
    tgt_block = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    parts2 = re.split(r"(?i)distractor grid\s*2\s*:\s*", rest)
    d1_block = parts2[0] if len(parts2)>0 else ""
    d2_block = parts2[1] if len(parts2)>1 else ""
    # trim headers
    tgt = _parse_grid_block(re.split(r"(?i)target grid\s*:\s*", tgt_block)[-1])
    d1 = _parse_grid_block(d1_block)
    d2 = _parse_grid_block(d2_block)
    return tgt, d1, d2

#remembers full/no more space rows of grid
def _grids_features(grid):
    return {
        "rows_full": _rows_full_of_X(grid),
        "cols_full": _cols_full_of_X(grid),
        "main_diag": _main_diag_full_X(grid),
        "anti_diag": _anti_diag_full_X(grid),
    }

#extracts contraints out of the explanation    
def _expression_constraints(expr: str):
    """Extract simple constraints like 'third row full', 'second column full'."""
    s = _norm(expr)
    constraints = []
    for m in re.finditer(r"(first|second|third|fourth|fifth|last|[1-5]|top|bottom|leftmost|rightmost|middle|center)\s+(row|column|col)", s):
        ord_tok, axis = m.group(1), "column" if "col" in m.group(2) else m.group(2)
        idx = ORD2IDX.get(ord_tok, None)
        if idx is None and ord_tok.isdigit(): idx = int(ord_tok)-1
        if idx is not None:
            constraints.append(("row_full" if axis=="row" else "col_full", idx))
    # diagonals
    if "main diagonal" in s or "primary diagonal" in s or "top-left to bottom-right" in s:
        constraints.append(("main_diag", True))
    if "anti diagonal" in s or "secondary diagonal" in s or "top-right to bottom-left" in s:
        constraints.append(("anti_diag", True))
    return constraints

def _satisfies(grid_feats, constraints):
    for kind, val in constraints:
        if kind=="row_full":
            if val not in grid_feats["rows_full"]: return False
        elif kind=="col_full":
            if val not in grid_feats["cols_full"]: return False
        elif kind=="main_diag":
            if not grid_feats["main_diag"]: return False
        elif kind=="anti_diag":
            if not grid_feats["anti_diag"]: return False
    return True

# ------------------------------ TABOO helpers ------------------------------

def _parse_taboo_from_prompt(prompt_text: str):
    """
    Robustly extract target word and related words from the Taboo prompt.
    - Accepts presence/absence of a colon after the 'target word' line.
    - Accepts 'Related words are:' or 'Related words:'.
    - Never crashes on empty/malformed input.
    """
    # 1) Target block: everything between "...target word..." and "Related words..."
    m = re.search(
        r"This is the target word.*?(?:guess:|:)?\s*([\s\S]*?)\n\nRelated words",
        prompt_text,
        re.I,
    )
    target_block = m.group(1) if m else ""
    # fallback: take the last non-empty line before "Related words"
    if not target_block.strip():
        pre = re.split(r"(?i)Related words", prompt_text)[0]
        lines = [ln.strip() for ln in pre.splitlines()]
        for ln in reversed(lines):
            if ln and "target word" not in ln.lower():
                target_block = ln
                break

    # first non-empty line is the target (if any)
    target = ""
    for ln in target_block.splitlines():
        ln = ln.strip()
        if ln:
            target = ln.strip().strip(".")
            break

    # 2) Related words block
    rel_m = re.search(
        r"Related words(?:\s+are)?\s*:\s*([\s\S]*?)\n\nImportant",
        prompt_text,
        re.I,
    )
    related = []
    if rel_m:
        for line in rel_m.group(1).splitlines():
            mm = re.search(r"-\s*(\S+)", line)
            if mm:
                related.append(mm.group(1).strip())

    return target, related


def _badword_hit(clue: str, banned: List[str]) -> bool:
    """Return True if clue uses the target or any related word (incl. simple variants)."""
    clue_l = clue.lower()
    toks = _tokenize(clue)
    for w in banned:
        w = w.lower()
        # exaktes Token
        if w in toks:
            return True
        # einfache Morphologie / Genitiv / Plural / Ableitungen
        if re.search(rf"\b{re.escape(w)}(?:'s|s|ed|ing|er|ers|ly)?\b", clue_l):
            return True
        # Bindestrich-Varianten
        if w in clue_l.replace("-", " "):
            return True
    return False



#-----------------------------REWARD FUNCTIONS------------------------------------

# ------------------------------ 1) IMAGEGAME ------------------------------

"""
Idea: We simulate what the model’s Instruction: lines would do to a start grid and measure the progress toward a target grid.

How it works:

Target grid is taken from kwargs["target_grid"] or parsed from the prompt (the block at the end of the imagegame prompt).
Start grid defaults to empty (or kwargs["start_grid"] if provided).

We collect all lines matching Instruction: … from the model output.

For each instruction, we apply it to a working copy of the grid using _apply_instruction, which supports:

placing a single letter at a (row, column) location, e.g.
“Put an E in second row third column”

filling a whole row/column with a letter, e.g.
“Fill the fifth row with X”

We compute before/after matches against the target: number of identical cells (0..25).

Scoring:

Format bonus: +0.1 if at least one Instruction: is present.

Progress: +(matches_after − matches_before)/25.

DONE handling: If the output contains DONE:

+0.2 only if the grid exactly equals the target;

otherwise -0.2 (premature completion).

Final reward is clipped to [-1, 1].

Example: If the grid improves from 10 to 18 correct cells, delta = 8 → 8/25 ≈ 0.32. With format +0.1, total ≈ 0.42. If DONE and exact match, add +0.2 → 0.62.

Limitations: _apply_instruction covers the most common phrasing patterns (place single cell; fill full row/column). Freer language may be ignored (no error, just no progress).
"""


def reward_imagegame(completions: List[Any], **kwargs) -> List[float]:
    """
    Reward = normalized improvement toward target grid by executing the model's "Instruction:" lines.
      + format bonuses for correct "Instruction:" usage
      + correctness for 'DONE' only when grid fully matches target
    Needs target grid; will parse from prompt if possible. If not found, falls back to format reward.
    Optional kwargs:
      - target_grid: str (5x5 text)
      - start_grid: str (default = empty grid)
    """
    prompts = _get_prompt_texts(kwargs)
    target_grid_text = kwargs.get("target_grid")
    start_grid_text = kwargs.get("start_grid")

    # Try to infer target grid from imagegame prompt if missing
    if not target_grid_text and prompts:
        # Use the last grid block in the prompt (after "Ok. Please do this for the following example")
        pm = re.split(r"Ok\.\s*Please do this.*?\n", prompts[0], flags=re.I|re.S)
        if len(pm) > 1:
            target_grid_text = pm[-1]
        else:
            # else: grab the last 5 lines containing '▢' or letters
            lines = [ln for ln in prompts[0].splitlines() if "▢" in ln or re.search(r"[A-Za-z]", ln)]
            target_grid_text = "\n".join(lines[-5:]) if len(lines) >= 5 else None

    tgt = _parse_grid_block(target_grid_text) if target_grid_text else None
    cur = _parse_grid_block(start_grid_text) if start_grid_text else _empty_grid()

    rewards = []
    for i, comp in enumerate(completions):
        text = _as_text(comp)
        done_declared = bool(re.search(r"(?i)\bDONE\b", text))
        instrs = re.findall(r"(?im)^\s*Instruction:\s*(.+)$", text)
        if not instrs and not tgt:
            # no target => only format compliance
            rewards.append(0.2 if _contains(text, "Instruction:") else 0.0)
            continue

        local_tgt = tgt
        # (allow per-sample prompt override if provided)
        if not local_tgt and prompts:
            # best effort per-sample
            local_tgt = _empty_grid()

        before = _grid_match_count(cur, local_tgt) if local_tgt else 0
        grid_tmp = [row[:] for row in cur]

        for ins in instrs:
            _apply_instruction(grid_tmp, ins)

        after = _grid_match_count(grid_tmp, local_tgt) if local_tgt else before
        delta = after - before
        # normalize by 25 cells, plus format bonus for valid instructions
        r = 0.0
        if instrs:
            r += 0.1
        r += (delta / 25.0)
        if done_declared:
            if local_tgt and _grid_equal(grid_tmp, local_tgt):
                r += 0.2
            else:
                r -= 0.2  # premature DONE
        rewards.append(_clip(r))
    return rewards

# ------------------------------ 2) REFERENCEGAME ------------------------------

"""
Idea: Reward a referring expression that uniquely identifies the target grid against two distractors, using robust, easily parsed features: “row full of X”, “column full of X”, “main/anti diagonal full of X”.

How it works:

Parse the target and two distractor grids from the prompt (or from kwargs).

Strip the leading tag and read the expression after Expression:.

Extract constraints with _expression_constraints, e.g.:

("row_full", 2) for “third row full”

("col_full", 1) for “second column full”

("main_diag", True), ("anti_diag", True)

Compute, for each grid, whether it satisfies all constraints.

Scoring:

Tag bonus: +0.2 if the output starts with Expression:.

Uniqueness:

+0.8 if only the target satisfies the constraints;

+0.3 if constraints are ambiguous (>=2 grids match) but the target is among them;

0.0 otherwise.

Clip to [-1, 1].

Example: Target has the third row fully X; distractors have first and second rows fully X. Expression “third row is fully X” → unique → 0.2 + 0.8 = 1.0.

Limitations: We score “full rows/columns/diagonals.” For more complex shapes (L-shapes, borders, counts), extend both the constraint extractor and the feature set.

"""

def reward_referencegame(completions: List[Any], **kwargs) -> List[float]:
    """
    Reward if 'Expression: ...' uniquely identifies the target grid among two distractors.
    Parsing: tries to read the 3 grids from the prompt (or kwargs: target_grid, distractor1, distractor2).
    Constraints extracted: 'third row', 'second column', diagonals. We check 'full of X'.
    Rewards:
      +0.2 correct tag 'Expression:'
      +0.8 if constraints select only the target; +0.3 if ambiguous but includes a *true* target feature; 0 otherwise.
    """
    prompts = _get_prompt_texts(kwargs)
    t_text = kwargs.get("target_grid")
    d1_text = kwargs.get("distractor1")
    d2_text = kwargs.get("distractor2")

    if (not t_text or not d1_text or not d2_text) and prompts:
        try:
            t, d1, d2 = _parse_three_grids_from_prompt(prompts[0])
        except Exception:
            t = d1 = d2 = None
    else:
        t = _parse_grid_block(t_text) if t_text else None
        d1 = _parse_grid_block(d1_text) if d1_text else None
        d2 = _parse_grid_block(d2_text) if d2_text else None

    rewards = []
    for comp in completions:
        text = _as_text(comp)
        tag_bonus = 0.2 if _starts_with_tag(text, "expression:") else 0.0
        expr = re.sub(r"(?i)^expression:\s*", "", text).strip()

        if not (t and d1 and d2):
            # no grids available: only format reward
            rewards.append(tag_bonus)
            continue

        feats = [_grids_features(g) for g in (t, d1, d2)]
        constraints = _expression_constraints(expr)

        # If no explicit constraint extracted, try a simple heuristic for rows 'full of X' mentioned as "row 3 all X"
        if not constraints:
            m = re.search(r"(first|second|third|fourth|fifth|last|[1-5]).*?(row).*?(all|full).*?x", _norm(expr))
            if m:
                idx = ORD2IDX.get(m.group(1), None)
                if idx is None and m.group(1).isdigit(): idx = int(m.group(1))-1
                if idx is not None:
                    constraints = [("row_full", idx)]

        matches = [ _satisfies(f, constraints) for f in feats ] if constraints else [False, False, False]

        score = 0.0
        if any(matches):
            if matches == [True, False, False]:
                score = 0.8
            elif matches.count(True) == 1 and matches[0]:
                score = 0.8
            elif matches.count(True) > 1:
                # ambiguous, but maybe constraints do match a target-only feature?
                # award small partial if target does match at least one true feature that distractors lack
                score = 0.3
        rewards.append(_clip(tag_bonus + score))
    return rewards

# ------------------------------ 3) TABOO ------------------------------

"""
Idea: Reward exactly one CLUE: … line that does not use prohibited words (the target and related words, including simple morphological variants), and that is short yet meaningful.

How it works:

Parse target and related words from the prompt (or use kwargs["target"], kwargs["related_words"]).

Extract exactly one CLUE: line.

Check for banned words and simple variations (possessive 's, plural s, -ed, -ing, hyphenated forms, etc.).

Scoring:

+0.3 if the format is correct (CLUE: at line start).

+0.4 if no banned words (else -0.4).

+0.2 for brevity & substance: 2–12 tokens and at least 2 content words.

-0.3 if there are multiple CLUE: lines or if GUESS: appears (extra chatter).

Clip to [-1, 1].

Example:
CLUE: opposite of yes
Format +0.3, no taboo hits +0.4, 3 tokens with ≥2 content words +0.2 → 0.9 total.

Edge cases: The morphology check is lightweight; for heavy-inflection languages, you may want a stronger normalizer.
"""

def reward_taboo(completions: List[Any], **kwargs) -> List[float]:
    """
    Rewards a *single* CLUE line that obeys taboo rules and is concise.
    Sources:
      - target, related words parsed from prompt, or kwargs: target, related_words (list[str])
    Scoring:
      +0.3 correct format 'CLUE: ...'
      +0.4 safe: no target/related words (incl. simple morphological variants)
      +0.2 brevity & substance: 2..12 words and >=2 content words
      -0.3 if includes 'GUESS:' or multiple clues or extra chatter
    """
    prompts = _get_prompt_texts(kwargs)
    target = kwargs.get("target")
    related = kwargs.get("related_words")

    if (not target or not related) and prompts:
        tgt, rel = _parse_taboo_from_prompt(prompts[0])
        target = target or tgt
        related = related or rel or []

    banned = [w for w in ([target] + (related or [])) if w]

    rewards = []
    for comp in completions:
        text = _as_text(comp)
        clue_lines = re.findall(r"(?im)^\s*CLUE:\s*(.+)$", text)
        r = 0.0
        if clue_lines:
            r += 0.3
            clue = clue_lines[0].strip()
            # safety
            if banned and not _badword_hit(clue, banned):
                r += 0.4
            elif banned:
                r -= 0.4
            # brevity + substance
            wc = len(_tokenize(clue))
            if 2 <= wc <= 12 and _count_content_words(clue) >= 2:
                r += 0.2
        else:
            # no 'CLUE:' prefix → zero by default
            r += 0.0

        # penalties
        if len(clue_lines) > 1 or _contains(text, "GUESS:"):
            r -= 0.3

        rewards.append(_clip(r))
    return rewards

# ------------------------------ 4) WORDLE (base) ------------------------------
"""
Idea: Reward correct format and the quality of the five-letter guess. Quality comes either from a known target (best supervision) or from provided feedback tokens when the target is unknown.

Scoring breakdown:

Format:

+0.2 if guess: with a 5-letter token is present.

+0.1 if explanation: is present.

+0.1 if the guess matches [a-z]{5} (formal validity).

-0.2 if the model fabricates a guess_feedback: line (forbidden).

Content (up to +0.6):

If target is provided:
_wordle_score_from_target(guess, target) where
score = (greens + 0.5 * yellows) / 5 in [0,1]. Multiply by 0.6.

Else if feedback is provided:
Parse a<yellow> p<green> ...; compute (greens + 0.5*yellows)/5 * 0.6.

Finally, clip to [-1, 1].

Broadcasting labels: If you pass a scalar target/feedback, it’s broadcast to the whole batch for convenience.

Example (with target “crane”):
Guess caper: suppose greens=2, yellows=1 → (2 + 0.5)/5 = 0.5; content 0.6*0.5=0.3.
Format (+0.2 +0.1 +0.1) → 0.4. Total ≈ 0.7 (assuming no fabricated feedback).
"""


def reward_wordle(completions: List[Any], **kwargs) -> List[float]:
    """
    Scores a single Wordle guess for:
      - correct format 'guess:' (+0.2) and 'explanation:' (+0.1)
      - valid 5-letter word (+0.1)
      - NO fabricated 'guess_feedback:' (−0.2)
      - semantic score from target or from provided feedback (greens 1.0, yellows 0.5; normalized to [0,1])
    Optional kwargs per sample (lists or scalars):
      - target: str (the secret word)  -> strongest supervision
      - feedback: str like 'a<yellow> p<green> ...' -> partial supervision
    Also accepts parsing the clue from prompt for the _withclue/_withcritic variants (but plain wordle ignores clue).
    """
    # broadcast scalar kwargs to per-sample lists
    def _as_list(v, n): return [v]*n if (v is not None and not isinstance(v, (list, tuple))) else (v or [None]*n)
    N = len(completions)

    targets = _as_list(kwargs.get("target"), N)
    feedbacks = _as_list(kwargs.get("feedback"), N)

    rewards = []
    for i, comp in enumerate(completions):
        text = _as_text(comp)
        guess = _extract_guess(text)
        expl  = _extract_explanation(text)

        r = 0.0
        if guess: r += 0.2
        if expl:  r += 0.1
        if guess and re.fullmatch(r"[a-z]{5}", guess): r += 0.1
        if _has_fabricated_feedback(text): r -= 0.2

        # target-based scoring
        if targets[i] and guess:
            r += 0.6 * _wordle_score_from_target(guess, targets[i].lower())
        # feedback-based scoring (we don't know the target)
        elif feedbacks[i] and guess:
            pairs = _parse_feedback_string(feedbacks[i])
            greens = sum(1 for (ch, col) in pairs if col == "green")
            yellows = sum(1 for (ch, col) in pairs if col == "yellow")
            # normalize by 5
            r += 0.6 * _clip((greens + 0.5*yellows)/5.0, 0.0, 1.0)

        rewards.append(_clip(r))
    return rewards

# ------------------------------ 5) WORDLE with CLUE ------------------------------
"""
Add-on to base Wordle: The explanation should actually use the clue.

How it works:

Parse the clue from the prompt (or kwargs["clue"]), tokenize to meaningful tokens.

Add +0.2 if the explanation contains at least one non-trivial clue token.

All the base Wordle scoring (format, content via target or feedback, anti-hallucination) still applies.

Example:
Clue: others’. If the explanation references possessives/ownership or literally includes a matching token, add +0.2.
"""


def reward_wordle_withclue(completions: List[Any], **kwargs) -> List[float]:
    """
    Same as reward_wordle, plus reward for USING the given clue in explanation:
      +0.2 if explanation references at least one non-stopword token from the clue.
    Clue is parsed from prompt if not in kwargs['clue'].
    """
    prompts = _get_prompt_texts(kwargs)
    clue = kwargs.get("clue")
    if not clue and prompts:
        m = re.search(r"(?im)^\s*clue\s*:\s*(.+)$", prompts[0])
        clue = m.group(1).strip() if m else None

    base = reward_wordle(completions, **kwargs)  # includes target/feedback handling if provided
    clue_tokens = set(t for t in _tokenize(clue or "") if len(t) > 2)

    out = []
    for r, comp in zip(base, completions):
        text = _as_text(comp)
        expl = _extract_explanation(text)
        bonus = 0.0
        if clue_tokens and any(tok in _tokenize(expl) for tok in clue_tokens):
            bonus += 0.2
        out.append(_clip(r + bonus))
    return out

# ------------------------------ 6) WORDLE with CRITIC ------------------------------
"""
Add-on to “withclue”: Make the model reflect the critic’s agreement/disagreement and adapt its guess if the critic disagreed.

Inputs:

critic_agreement ∈ { 'agree', 'disagree', True, False } (scalar or per-sample).

prior_guess (optional) — the previous guess to detect adaptation.

Bonuses:

+0.15 if the explanation explicitly references agree/disagree in line with critic_agreement.

+0.10 if critic_agreement == 'disagree' and the new guess differs from prior_guess.

Plus the full reward from reward_wordle_withclue.

Example:
Critic: “disagree”, prior guess apple, new guess angle. Explanation mentions disagreement → +0.15; changed guess → +0.10. Add to the with-clue/base score.
"""


def reward_wordle_withcritic(completions: List[Any], **kwargs) -> List[float]:
    """
    Wordle + critic-awareness:
      - Base wordle reward (target/feedback if provided)
      - +0.15 if explanation explicitly references 'agree'/'disagree' consistent with kwargs['critic_agreement'] ∈ {'agree','disagree', True/False}
      - +0.10 if critic disagreed and the model changed guess compared to kwargs['prior_guess']
    Clue handling like wordle_withclue (optional).
    """
    base = reward_wordle_withclue(completions, **kwargs)

    def norm_agree(v) -> str | None:
        if isinstance(v, bool): return "agree" if v else "disagree"
        if isinstance(v, str):
            v = v.strip().lower()
            if v in {"agree","agreed"}: return "agree"
            if v in {"disagree","disagreed","reject"}: return "disagree"
        return None

    N = len(completions)
    critic = kwargs.get("critic_agreement")
    critic_list = [norm_agree(critic)]*N if not isinstance(critic, (list,tuple)) else [norm_agree(x) for x in critic]
    prior_guess = kwargs.get("prior_guess")
    prior_list = [prior_guess]*N if not isinstance(prior_guess, (list,tuple)) else list(prior_guess)

    out = []
    for i, (r, comp) in enumerate(zip(base, completions)):
        text = _as_text(comp)
        expl = _extract_explanation(text)
        guess = _extract_guess(text)
        bonus = 0.0

        if critic_list[i]:
            wants = critic_list[i]
            if wants == "agree" and re.search(r"(?i)\bagree\b", expl):
                bonus += 0.15
            if wants == "disagree" and re.search(r"(?i)\bdisagree|\bnot\b\s+agree", expl):
                bonus += 0.15

        if critic_list[i] == "disagree" and prior_list[i]:
            # reward adapting the guess
            if guess and guess != str(prior_list[i]).lower():
                bonus += 0.10

        out.append(_clip(r + bonus))
    return out

# ------------------------------ wiring helper ------------------------------
"""
A tiny convenience map from a game name string to the corresponding reward function:
{'imagegame','referencegame','taboo','wordle','wordle_withclue','wordle_withcritic'}
"""

def get_reward_fn_by_game(name: str):
    """
    Convenience selector.
    name ∈ {'imagegame','referencegame','taboo','wordle','wordle_withclue','wordle_withcritic'}
    """
    name = (name or "").lower()
    return {
        "imagegame": reward_imagegame,
        "referencegame": reward_referencegame,
        "taboo": reward_taboo,
        "wordle": reward_wordle,
        "wordle_withclue": reward_wordle_withclue,
        "wordle_withcritic": reward_wordle_withcritic,
    }.get(name)


# In[ ]:





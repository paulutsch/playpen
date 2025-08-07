# scorers/wordle_scorer.py
class WordleScorer:
    def score_turn(self, prefix_messages, response: str) -> float:
        """
        prefix_messages: List[{"role": str, "content": str}]
        response       : The model's latest reply (string).

        Returns a number in [0, 100].
        """
        # 1) extract the model's guess from `response`
        # 2) compare to hidden target word stored in self.state
        # 3) compute closeness / strategy score
        return score

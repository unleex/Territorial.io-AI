from pathlib import Path


class EloRating:
    """
    Standard ELO rating system adapted for team-vs-team matchups.

    Each registered model gets a rating. After every game:
      expected_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
      R_A += K * (score_A - expected_A)   where score: win=1, draw=0.5, loss=0

    K controls how fast ratings move:
      - High K (e.g. 64): fast, good for early calibration with few games
      - Low K  (e.g. 16): stable, good for fine-grained comparison
    """

    def __init__(self, k: float = 32, initial: float = 1000.0):
        self.k = k
        self.initial = initial
        self.ratings: dict[str, float] = {}
        self.games_played: dict[str, int] = {}

    def register(self, name: str):
        if name not in self.ratings:
            self.ratings[name] = self.initial
            self.games_played[name] = 0

    def expected(self, name_a: str, name_b: str) -> float:
        """Probability that A beats B according to current ratings."""
        return 1.0 / (1.0 + 10 ** ((self.ratings[name_b] - self.ratings[name_a]) / 400))

    def update_from_batch(
        self, name_a: str, name_b: str, empirical_win_rate: float, custom_k=None
    ):
        """
        Updates ratings using an aggregated win rate (0.0 to 1.0) instead of individual rows.
        empirical_win_rate: The mean score of name_a against name_b (e.g., 0.75 if name_a won 75%).
        """
        k = custom_k if custom_k is not None else self.k

        # 1. Calculate expected win probabilities based on current frozen ratings
        e_a = self.expected(name_a, name_b)
        e_b = 1.0 - e_a

        # 2. Policy B's empirical win rate is simply the inverse of Policy A's
        empirical_win_rate_b = 1.0 - empirical_win_rate

        # 3. Compute deltas directly using the fractional scores
        delta_a = k * (empirical_win_rate - e_a)
        delta_b = k * (empirical_win_rate_b - e_b)

        # 4. Mutate global states synchronously
        self.ratings[name_a] += delta_a
        self.ratings[name_b] += delta_b

        # We treat this entire batch as 1 macro-match step for tracking history
        self.games_played[name_a] += 1
        self.games_played[name_b] += 1

    def summary(self) -> str:
        rows = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        lines = [
            "=" * 56,
            f"{'ELO RANKINGS':^56}",
            "=" * 56,
            f"  {'Model':<36} {'ELO':>6}  {'Games':>5}",
            "-" * 56,
        ]
        for name, rating in rows:
            short = Path(name).parent.name if "/" in name else name
            lines.append(f"  {short:<36} {rating:>6.1f}  {self.games_played[name]:>5}")
        lines.append("=" * 56)
        return "\n".join(lines)

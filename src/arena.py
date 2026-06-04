import os
from configs.mac_config import config
from pathlib import Path
from itertools import combinations
import imageio.v2 as imageio
import numpy as np
import model as _model_module
from environment import CustomEnvironment
from game.game import Game
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog


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

    def update(self, name_a: str, name_b: str, score_a: float):
        """
        Update ratings after one game.
        score_a: 1.0 = A won, 0.5 = draw, 0.0 = B won.
        """
        e_a = self.expected(name_a, name_b)
        e_b = 1.0 - e_a
        self.ratings[name_a] += self.k * (score_a - e_a)
        self.ratings[name_b] += self.k * ((1.0 - score_a) - e_b)
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


def _load_algo(checkpoint_path: str):
    eval_config = (
        config
        .resources(num_gpus=0)
        .env_runners(num_env_runners=0)
    )
    
    algo = eval_config.build_algo()
    algo.restore(checkpoint_path)
    return algo


def _get_action(algo_or_policy, obs: dict):
    if isinstance(algo_or_policy, Policy):
        action, _, _ = algo_or_policy.compute_single_action(obs, explore=False)
        return action
    return algo_or_policy.compute_single_action(obs, policy_id="p0", explore=False)


class ArenaEnvironment(CustomEnvironment):
    def _prepare(self):
        super()._prepare()
        self.game.agents = list(range(self.game.n_players))


def _run_matchup(
    algo_a: PPO,
    algo_b: PPO,
    name_a: str,
    name_b: str,
    num_games: int,
    team_a_ids: list[int],
    team_b_ids: list[int],
    elo: EloRating,
    video_dir: str | None,
):
    """
    Play num_games between algo_a and algo_b, update ELO after each game.
    Returns (wins_a, wins_b, draws).
    """
    wins = {"A": 0, "B": 0, "draw": 0}
    env = ArenaEnvironment(rendering=False)

    for ep in range(num_games):
        observations, _ = env.reset()

        cumulative = {agent: 0.0 for agent in team_a_ids + team_b_ids}

        writer = None
        if video_dir:
            tag = f"{Path(name_a).parent.name}_vs_{Path(name_b).parent.name}"
            writer = imageio.get_writer(
                str(Path(video_dir) / f"{tag}_ep{ep:03d}.mp4"), fps=8
            )

        while env.agents:
            actions = {}
            for agent in env.agents:
                algo = algo_a if agent in team_a_ids else algo_b
                actions[agent] = _get_action(algo, observations[agent])

            observations, step_rewards, terminations, truncations, _ = env.step(actions)

            for agent, r in step_rewards.items():
                cumulative[agent] += r

            if writer is not None:
                # render frame here if you add a renderer call

                pass  # video rendering requires a render call; add if needed

        if writer is not None:
            writer.close()

        final_sizes = {
            agent: (
                env.game.id_to_country[agent].size
                if agent in env.game.id_to_country
                else 0
            )
            for agent in team_a_ids + team_b_ids
        }
        mean_a = np.mean([final_sizes[a] for a in team_a_ids])
        mean_b = np.mean([final_sizes[b] for b in team_b_ids])

        if mean_a > mean_b:
            score_a, result = 1.0, "A"
        elif mean_b > mean_a:
            score_a, result = 0.0, "B"
        else:
            score_a, result = 0.5, "draw"

        wins[result if result != "draw" else "draw"] += 1
        elo.update(name_a, name_b, score_a)

        short_a = Path(name_a).parent.name
        short_b = Path(name_b).parent.name
        print(
            f"  [{short_a} vs {short_b}] game {ep + 1:>3}/{num_games}"
            f"  winner={'A' if result == 'A' else ('B' if result == 'B' else 'draw')}"
            f"  tiles A={mean_a:5.1f}  B={mean_b:5.1f}"
            f"  ELO → {name_a}: {elo.ratings[name_a]:.1f}"
            f"  {name_b}: {elo.ratings[name_b]:.1f}"
        )

    return wins


def run_arena(
    checkpoint_agent: str,
    checkpoint_bots: str,
    num_episodes: int = 100,
    agents_idxs: list[int] | None = None,
    bots_idxs: list[int] | None = None,
    video_dir: str | None = None,
    elo_k: float = 32,
) -> EloRating:
    """
    Head-to-head evaluation between two models with ELO tracking.

    Returns the EloRating object so you can inspect .ratings afterwards.
    To compare more than two models use run_tournament().
    """
    team_a = agents_idxs or [0, 1, 2, 3]
    team_b = bots_idxs or [4, 5, 6, 7]

    if video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    elo = EloRating(k=elo_k)
    elo.register(checkpoint_agent)
    elo.register(checkpoint_bots)

    print(f"\n[arena] Loading model A: {Path(checkpoint_agent).parent.name}")
    algo_a = _load_algo(checkpoint_agent)
    print(f"[arena] Loading model B: {Path(checkpoint_bots).parent.name}")
    algo_b = _load_algo(checkpoint_bots)

    wins = _run_matchup(
        algo_a,
        algo_b,
        checkpoint_agent,
        checkpoint_bots,
        num_episodes,
        team_a,
        team_b,
        elo,
        video_dir,
    )

    print(f"\n  Wins — A: {wins['A']}  B: {wins['B']}  Draws: {wins['draw']}")
    print(elo.summary())
    return elo


def run_tournament(
    checkpoints: list[str],
    num_games_per_matchup: int = 20,
    team_size: int = 4,
    video_dir: str | None = None,
    elo_k: float = 32,
) -> EloRating:
    """
    Round-robin tournament across any number of models.
    Each pair plays num_games_per_matchup games.
    Players 0..team_size-1 are team A, team_size..2*team_size-1 are team B.

    Example:
        elo = run_tournament([
            "logs/.../checkpoint_A",
            "logs/.../checkpoint_B",
            "logs/.../checkpoint_C",
        ], num_games_per_matchup=20)
    """
    if video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    team_a_ids = list(range(team_size))
    team_b_ids = list(range(team_size, team_size * 2))

    elo = EloRating(k=elo_k)
    for cp in checkpoints:
        elo.register(cp)

    algos = {}
    for cp in checkpoints:
        print(f"[tournament] Loading {Path(cp).parent.name}...")
        algos[cp] = _load_algo(cp)

    pairs = list(combinations(checkpoints, 2))
    print(
        f"\n[tournament] {len(checkpoints)} models, {len(pairs)} matchups"
        f", {num_games_per_matchup} games each"
        f", {len(pairs) * num_games_per_matchup} total games\n"
    )

    for i, (cp_a, cp_b) in enumerate(pairs, 1):
        print(
            f"── Matchup {i}/{len(pairs)}: "
            f"{Path(cp_a).parent.name} vs {Path(cp_b).parent.name} ──"
        )
        wins = _run_matchup(
            algos[cp_a],
            algos[cp_b],
            cp_a,
            cp_b,
            num_games_per_matchup,
            team_a_ids,
            team_b_ids,
            elo,
            video_dir,
        )
        print(f"  Wins — A: {wins['A']}  B: {wins['B']}  Draws: {wins['draw']}\n")

    print(elo.summary())
    return elo


if __name__ == "__main__":
    run_arena(
        checkpoint_agent="/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/all_pretrained_agents/PPO_custom_env_8d4b7_00000_0_2026-06-02_19-31-48/checkpoint_000007",
        checkpoint_bots="/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/Multiagency/best_bots_only/checkpoint_000012",
        num_episodes=100,
        agents_idxs=[0, 1, 2, 3],
        bots_idxs=[4, 5, 6, 7],
        video_dir=str(Path("logs") / "arena_validation" / "videos"),
    )

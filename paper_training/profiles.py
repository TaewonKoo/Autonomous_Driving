"""Runnable paper configurations. Local choices remain visible in config.json."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    paper: int
    source: str
    algorithm: str
    scenario: str
    horizon: int
    iterations: int = 0
    steps: int = 0
    rollouts: int = 1
    seeds: tuple = (23,)
    dt: float = .1
    actor: tuple = (64, 64)
    critic: tuple = (64, 64)
    activation: str = "tanh"
    lr: float = 3e-4
    batch: int = 64
    epochs: int = 10
    gamma: float = .99
    gae: float = .95
    clip: float = .2
    kl: float = 0.
    anneal: bool = False
    evaluation: int = 10
    evaluation_horizon: int = 0
    cavs: int = 1
    network: str = "mlp"
    cooperation: str = "global"
    radius: int = 1
    paper_settings: dict = field(default_factory=dict)
    local_choices: str = ""


PROFILES = {
    "lane-ppo": Profile(
        1, "https://arxiv.org/html/2002.02667v2", "ppo", "lane", 512,
        iterations=500, lr=1e-4, anneal=True, evaluation=50, evaluation_horizon=1024,
        paper_settings=dict(accel=2.5, brake=-4.5, near_m=10, collision=-100),
        local_choices="Seongsan lane changes; 500 updates, one sampler, linear annealing, 64x64 tanh, "
                      "zero entropy bonus and reward weights are local choices."),
    "benchmark-merge-ppo": Profile(
        2, "https://proceedings.mlr.press/v87/vinitsky18a/vinitsky18a.pdf",
        "ppo", "benchmark", 750, iterations=500, rollouts=50, seeds=(23, 24, 25),
        critic=(256, 256), lr=5e-4, gae=.97, cavs=5,
        paper_settings=dict(variant="merge0", nominal_penetration=.10),
        local_choices="Merge0 transfer; fixed 5 slots on Seongsan, approximately 10% background CAVs "
                      "plus ego. Actor 64x64 tanh, gamma, clip, minibatch, reward parameters and "
                      "seed numbers are local choices; historical RLlib defaults were not assumed."),
    "flow-trpo": Profile(
        3, "https://arxiv.org/html/1710.05465v4", "trpo", "flow", 3000,
        steps=6_000_000, actor=(3, 3), critic=(), gamma=.999, kl=.01,
        paper_settings=dict(vehicles=22, warmup=750, train_m=[220, 270], eval_m=[210, 290]),
        local_choices="Microscopic NumPy IDM ring approximates Flow dynamics. Rollout batch = one "
                      "trajectory, advantages use discounted returns and a linear feature baseline. "
                      "TRPO damping/CG/line search and action bounds are local choices."),
    "mujoco-ppo-transfer": Profile(
        4, "https://arxiv.org/pdf/1707.06347", "ppo", "transfer", 2048,
        steps=1_000_000,
        local_choices="MuJoCo Table 3 optimizer settings applied to Seongsan following control with "
                      "separate policy/value networks. Driving observations, reward, horizon within "
                      "the map, and one sampler are local choices. Roboschool and Atari are distinct tasks."),
    "cooperative-ppo": Profile(
        5, "https://www.eecis.udel.edu/lena/files/vnc19_lena.pdf", "ppo", "cooperative", 2000,
        iterations=200, rollouts=20, clip=.3, cavs=3,
        paper_settings=dict(ring_m=230, humans=30, cav_counts=[3, 6, 9], speed_target=25,
                            safe_headway_s=1, delta_speed=[-1, 1], table_lambda=.999),
        local_choices="NumPy IDM ring, default 3 CAVs and global maximum reward (Eq. 9). "
                      "Table I lambda is recorded; its role is ambiguous. LR, minibatch, epochs, "
                      "networks, gamma/GAE and step interval are local choices."),
    "intersection-kl-ppo": Profile(
        6, "https://www.mdpi.com/2076-3417/10/16/5722", "kl_ppo", "intersection", 600,
        iterations=200, rollouts=10, actor=(256, 256, 256), critic=(256, 256, 256),
        kl=.01, cavs=3, paper_settings=dict(desired_speed=15),
        local_choices="Seongsan north unsignalized approaches and nearby traffic; three control slots "
                      "with a shared vehicle policy, 0.1 s, Adam LR, minibatch, tanh and initial "
                      "KL penalty coefficient 1 are local choices except the 0.1 s paper interval."),
    "randomized-merge-sac": Profile(
        7, "https://www.mdpi.com/2075-1702/12/4/264", "sac", "merge", 1000,
        steps=200_000, actor=(256, 256), critic=(256, 256), batch=256,
        activation="relu", evaluation=1000,
        paper_settings=dict(accel=2.5, brake=-4.5, vehicle_length=5,
                            merge_weight=-.015, braking_weight=-.015, jerk_weight=-.015,
                            speed_diff_max=5, jerk_max=3, stop_penalty=-.5,
                            success=1, collision=-1),
        local_choices="Seongsan east slip merge; 1000-step cutoff, optimizer, network and replay "
                      "settings are local choices. Uses the randomized rule-based traffic arm."),
    "randomized-freeway-pasac": Profile(
        7, "https://www.mdpi.com/2075-1702/12/4/264", "pasac", "freeway", 1000,
        steps=400_000, actor=(256, 256), critic=(256, 256), batch=256,
        activation="relu", evaluation=1000,
        paper_settings=dict(accel=2.6, brake=-4.5, safe_speed=16.89, stable_speed=8.89,
                            safe_gap=25, extra_gap=2.5, collision=-200,
                            weights=[-5, -2, -10, -.005, 1, -.5, -.5]),
        local_choices="Seongsan multi-lane segments; the policy produces lane scores with argmax "
                      "and a continuous acceleration. Critics use executed one-hot lane decisions "
                      "with a straight-through gradient for the actor. Cutoff, network and "
                      "optimizer/replay settings are local choices."),
}

TRAFFIC_RANDOMIZATION = {
    "delta": (3.5, 4.5), "tau": (.5, 1.5), "accel": (1.8, 3.4),
    "decel": (3.5, 5.5), "maxSpeed": (7.33, 9.33), "lcSpeedGain": (0, 100),
    "lcAssertive": (1, 5),
}

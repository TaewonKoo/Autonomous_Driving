"""Resolve the declared learning-only PPO candidates without changing the task."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .observed_driving import PPO_PROFILE


def resolve_candidate(path, candidate_id):
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"))
    protocol = document["common_protocol"]
    if candidate_id == protocol["control_id"]:
        definition = {"id": candidate_id, "name": "동일 예산 기준 설정", "overrides": {},
                      "effective_learning_settings": protocol["control_settings"]}
    else:
        definitions = [c for c in document["candidates"] if c["id"] == candidate_id]
        if len(definitions) != 1:
            raise ValueError(f"Candidate id must occur exactly once: {candidate_id}")
        definition = definitions[0]
    s = definition["effective_learning_settings"]
    fixed = {"gamma": .99, "gae_lambda": .95, "total_samples_per_update": 2048,
             "worker_count": 4, "samples_per_worker": 512, "adam_epsilon": 1e-8,
             "adam_betas": [.9, .999], "entropy_bonus": 0., "advantage_normalization": True,
             "value_loss": "separate optimizer, MSE, no value clipping", "gradient_clipping": False}
    for key, value in fixed.items():
        if s.get(key) != value:
            raise ValueError(f"This sweep keeps {key} fixed at {value!r}")
    allowed = {"learning_rate": (3e-5, 1e-4, 3e-4), "learning_rate_schedule": ("constant", "linear_to_zero"),
               "clip": (.1, .2, .3), "epochs": (3, 10), "minibatch_size": (64, 128, 256)}
    for key, values in allowed.items():
        if s[key] not in values:
            raise ValueError(f"Outside the declared literature-backed search space: {key}={s[key]}")
    if set(s) != set(fixed) | set(allowed):
        raise ValueError("Unrecognized or missing learning-setting keys; refusing to silently ignore them")
    profile = replace(PPO_PROFILE, lr=s["learning_rate"], anneal=s["learning_rate_schedule"] == "linear_to_zero",
                      clip=s["clip"], epochs=s["epochs"], batch=s["minibatch_size"])
    return profile, {"definition": deepcopy(definition), "manifest_path": str(path),
                     "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                     "initial_checkpoint_sha256": protocol["initial_checkpoint_sha256"],
                     "evaluation": protocol["evaluation"],
                     "scope": "Seongsan tuning using published scalar values; not whole-paper replication"}

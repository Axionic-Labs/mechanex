from _example_utils import get_remote_client


def main():
    mx = get_remote_client()

    policy = {
        "name": "remote_smoke_policy",
        "sampling": {"method": "speculative-decoding", "draft_model": "default", "temperature": 0.2},
        "constraints": {"json_mode": True, "required_fields": ["answer"]},
        "verifiers": {"enabled": ["syntax"], "repair_on_failure": True},
        "optimization": {"best_of_n": 1, "fallback_methods": ["top-k"]},
    }

    policy_id = mx.policy.save(policy)
    print("Saved policy:", policy_id)
    out = mx.policy.run(
        prompt="Return JSON with one field answer set to ok.",
        policy_id=policy_id,
        include_trace=True,
    )
    print("Output:", out["output"])
    print("Accepted:", out["accepted"])


if __name__ == "__main__":
    main()

from _example_utils import get_local_client


def main():
    mx = get_local_client()

    policy = {
        "name": "local_strict_json",
        "sampling": {"method": "guided-generation", "temperature": 0.2, "top_p": 0.9},
        "constraints": {
            "json_mode": True,
            "json_schema": {
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            },
        },
        "verifiers": {"enabled": ["syntax", "json_schema"], "repair_on_failure": True},
        "optimization": {"best_of_n": 1},
    }

    result = mx.policy.run(
        prompt="Return JSON only with one field named answer with value ok.",
        policy=policy,
        include_trace=True,
    )
    print("Output:", result["output"])
    print("Accepted:", result["accepted"])
    if result.get("trace"):
        print("Activated controls:", result["trace"].get("policy_name"))


if __name__ == "__main__":
    main()

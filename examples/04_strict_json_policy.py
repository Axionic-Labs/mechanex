from _example_utils import get_remote_client


def main():
    mx = get_remote_client()
    schema = {
        "type": "object",
        "required": ["ticket_id", "priority", "summary"],
        "properties": {
            "ticket_id": {"type": "string"},
            "priority": {"type": "string"},
            "summary": {"type": "string"},
        },
    }
    policy = mx.policy.strict_json_extraction(schema=schema, name="strict_json_support_v1")
    policy_id = mx.policy.save(policy)
    print("Saved policy_id:", policy_id)

    result = mx.policy.run(
        prompt="Extract a support ticket in JSON. Ticket id: T-900. Priority: high. Summary: Login fails.",
        policy_id=policy_id,
        include_trace=True,
    )
    print("Output:", result["output"])
    print("Accepted:", result["accepted"])


if __name__ == "__main__":
    main()

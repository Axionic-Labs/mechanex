from _example_utils import get_remote_client


def main():
    mx = get_remote_client()

    strict = mx.policy.strict_json_extraction(
        schema={
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
        name="strict_json_eval",
    )
    fast = mx.policy.fast_tool_router(name="fast_router_eval")

    compare = mx.policy.compare(
        prompt="Return a JSON object with one field answer=ok.",
        policies=[strict, fast],
        max_new_tokens=64,
    )
    print("Compare result count:", len(compare["results"]))

    prompts = [
        "Return JSON with answer=yes",
        "Return JSON with answer=no",
        "Return JSON with answer=maybe",
    ]
    evaluation = mx.policy.evaluate(prompts=prompts, policy=strict, max_new_tokens=64)
    print("Success rate:", evaluation["success_rate"])
    print("Avg latency ms:", evaluation["avg_latency_ms"])


if __name__ == "__main__":
    main()

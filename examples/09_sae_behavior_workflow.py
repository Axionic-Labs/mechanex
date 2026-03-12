from _example_utils import get_local_client, load_jsonl


def main():
    mx = get_local_client()
    rows = load_jsonl("examples/data/sae_behavior_dataset.jsonl")
    prompts = [x["prompt"] for x in rows]
    positive = [x["positive_answer"] for x in rows]
    negative = [x["negative_answer"] for x in rows]

    behavior = mx.sae.create_behavior(
        behavior_name="concise_mode",
        prompts=prompts,
        positive_answers=positive,
        negative_answers=negative,
        description="Encourage concise style",
    )
    print("Behavior created:", behavior.get("behavior_name"))

    out = mx.sae.generate(
        prompt="Describe how to optimize model inference in 3 concise points.",
        behavior_names=["concise_mode"],
        max_new_tokens=80,
    )
    print(out)


if __name__ == "__main__":
    main()

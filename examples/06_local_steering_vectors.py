from _example_utils import get_local_client, load_jsonl


def main():
    mx = get_local_client()
    dataset = load_jsonl("examples/data/steering_dataset.jsonl")
    prompts = [x["prompt"] for x in dataset]
    positive = [x["positive_answer"] for x in dataset]
    negative = [x["negative_answer"] for x in dataset]

    vector_id = mx.steering.generate_vectors(
        prompts=prompts,
        positive_answers=positive,
        negative_answers=negative,
        method="caa",
    )
    print("Local vector id:", vector_id)

    out = mx.generation.generate(
        prompt="Tell me how to design a secure API, briefly.",
        sampling_method="top-k",
        steering_vector=vector_id,
        steering_strength=0.6,
        max_tokens=80,
    )
    print(out)


if __name__ == "__main__":
    main()

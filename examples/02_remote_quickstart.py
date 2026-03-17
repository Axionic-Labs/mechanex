from _example_utils import get_remote_client


def main():
    mx = get_remote_client()

    output = mx.generation.generate(
        prompt="Give a concise summary of why policy-based inference helps small models.",
        sampling_method="top-p",
        top_p=0.9,
        temperature=0.6,
        max_tokens=96,
    )
    print(output)


if __name__ == "__main__":
    main()

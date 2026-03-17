from _example_utils import get_remote_client


def main():
    mx = get_remote_client()
    prompt = "Return one sentence explaining speculative decoding."

    strategies = [
        {"sampling_method": "top-k", "top_k": 40},
        {"sampling_method": "top-p", "top_p": 0.9},
        {"sampling_method": "min-p", "min_p": 0.08},
        {"sampling_method": "typical", "typical_p": 0.92},
        {"sampling_method": "constrained-beam-search"},
        {"sampling_method": "speculative-decoding", "draft_model": "default"},
        {"sampling_method": "ssd", "draft_model": "default"},
        {"sampling_method": "guided-generation", "regex_pattern": r"^.*\.$"},
        {"sampling_method": "ensemble-sampling", "ensemble_models": ["default"]},
        {"sampling_method": "ads", "ads_subset_size": 8, "ads_beta": 0.3},
    ]

    for cfg in strategies:
        method = cfg["sampling_method"]
        print(f"\n=== {method} ===")
        try:
            out = mx.generation.generate(prompt=prompt, max_tokens=64, **cfg)
            print(out)
        except Exception as exc:
            print(f"Failed: {exc}")


if __name__ == "__main__":
    main()

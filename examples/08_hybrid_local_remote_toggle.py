from _example_utils import get_local_client, get_remote_client


def run_with_mode(mode: str):
    if mode == "remote":
        mx = get_remote_client()
    else:
        mx = get_local_client()

    mx.set_execution_mode(mode)
    result = mx.generation.generate(
        prompt=f"Explain runtime policies in one sentence ({mode} mode).",
        sampling_method="top-p",
        top_p=0.9,
        max_tokens=60,
    )
    print(f"[{mode}] {result}")


def main():
    for mode in ["local", "remote"]:
        try:
            run_with_mode(mode)
        except Exception as exc:
            print(f"{mode} failed: {exc}")


if __name__ == "__main__":
    main()

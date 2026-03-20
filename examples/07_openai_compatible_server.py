from _example_utils import get_local_client


def main():
    mx = get_local_client()
    # Starts local OpenAI-compatible API at http://0.0.0.0:8000 by default.
    # POST /v1/chat/completions and /v1/completions support `policy` and `policy_id`.
    mx.serve(host="0.0.0.0", port=8000, use_vllm=False)


if __name__ == "__main__":
    main()

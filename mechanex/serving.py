from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import uuid
import os

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

def create_app(client, model=None, vllm_engine=None, corrected_behaviors=None):
    from .client import Mechanex
    import types
    if isinstance(client, types.ModuleType):
        # If the module was passed instead of the instance, try to find the singleton
        import mechanex
        client = getattr(mechanex, "_mx", client)
        if isinstance(client, types.ModuleType):
            raise ValueError("The 'client' argument must be an instance of Mechanex, not a module.")

    app = FastAPI(title="Mechanex OpenAI Compatible Server")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
            messages = body.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""
            
            max_tokens = body.get("max_tokens", 100)
            temperature = body.get("temperature", 0.7)
            top_p = body.get("top_p", 1.0)
            
            # Mechanistic features from body
            steering_vector = body.get("steering_vector") or body.get("steering_vector_id")
            steering_strength = body.get("steering_strength", 0)
            
            # Merge global corrected behaviors with request behaviors
            request_behaviors = body.get("behavior_names") or []
            global_behaviors = corrected_behaviors or []
            # Deduplicate while preserving order if possible (or just use set if order doesn't matter much)
            behavior_names = list(set(request_behaviors + global_behaviors))
            
            auto_correct = body.get("auto_correct", False) # Default to false for API unless behavior_names present
            force_steering = body.get("force_steering")
            
            if vllm_engine:
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                outputs = vllm_engine.generate([prompt], sampling_params)
                output = outputs[0].outputs[0].text
            else:
                # Use Mechanex features if requested
                if behavior_names or force_steering:
                    output = client.sae.generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        behavior_names=behavior_names,
                        auto_correct=auto_correct or bool(behavior_names),
                        force_steering=force_steering
                    )
                else:
                    output = client.generation.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        steering_vector=steering_vector,
                        steering_strength=steering_strength
                    )
            
            # Format as OpenAI response
            response_id = f"chatcmpl-{uuid.uuid4()}"
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": client.model_name or model or "mechanex-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output if isinstance(output, str) else output.get("output", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()), # Mock
                    "completion_tokens": len(str(output).split()), # Mock
                    "total_tokens": 0 # Mock
                }
            }
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "server_error"}})

    @app.post("/v1/completions")
    async def completions(request: Request):
        try:
            body = await request.json()
            prompt = body.get("prompt", "")
            max_tokens = body.get("max_tokens", 100)
            
            # Mechanistic features from body
            steering_vector = body.get("steering_vector") or body.get("steering_vector_id")
            steering_strength = body.get("steering_strength", 0)
            
            # Merge global corrected behaviors with request behaviors
            request_behaviors = body.get("behavior_names") or []
            global_behaviors = corrected_behaviors or []
            behavior_names = list(set(request_behaviors + global_behaviors))
            
            auto_correct = body.get("auto_correct", False)
            force_steering = body.get("force_steering")
            
            if vllm_engine:
                sampling_params = SamplingParams(max_tokens=max_tokens)
                outputs = vllm_engine.generate([prompt], sampling_params)
                output = outputs[0].outputs[0].text
            else:
                if behavior_names or force_steering:
                    output = client.sae.generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        behavior_names=behavior_names,
                        auto_correct=auto_correct or bool(behavior_names),
                        force_steering=force_steering
                    )
                else:
                    output = client.generation.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        steering_vector=steering_vector,
                        steering_strength=steering_strength
                    )
            
            response_id = f"cmpl-{uuid.uuid4()}"
            return {
                "id": response_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": client.model_name or model or "mechanex-model",
                "choices": [
                    {
                        "text": output if isinstance(output, str) else output.get("output", ""),
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length"
                    }
                ]
            }
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "server_error"}})

    return app

def run_server(client, model=None, host="0.0.0.0", port=8000, use_vllm=False, corrected_behaviors=None):
    vllm_engine = None
    if use_vllm:
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'.")
        
        model_name = model or client.model_name
        if not model_name:
            raise ValueError("Model name must be provided to use vLLM.")
            
        print(f"Initializing vLLM engine with model: {model_name}")
        vllm_engine = LLM(model=model_name)

    app = create_app(client, model, vllm_engine=vllm_engine, corrected_behaviors=corrected_behaviors)
    print(f"Starting Mechanex OpenAI-compatible server on {host}:{port}")
    if corrected_behaviors:
        print(f"Simulating SAE corrections for: {corrected_behaviors}")
    uvicorn.run(app, host=host, port=port)

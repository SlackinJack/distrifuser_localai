# https://github.com/xdit-project/xDiT/blob/1c31746e2f903e791bc2a41a0bc23614958e46cd/comfyui-xdit/host.py

import argparse
import os
import time
import torch
import torch.distributed as dist
import pickle
import io
# import json
import logging
import base64
import torch.multiprocessing as mp

from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
from flask import Flask, request, jsonify
from distrifuser.utils import DistriConfig
from distrifuser.pipelines import DistriSDPipeline, DistriSDXLPipeline

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

app = Flask(__name__)

os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

pipe = None
engine_config = None
input_config = None
local_rank = None
logger = None
initialized = False


def get_scheduler(scheduler_name, current_scheduler_config):
    scheduler_class = get_scheduler_class(scheduler_name)
    scheduler_config = get_scheduler_config(scheduler_name, current_scheduler_config)
    return scheduler_class.from_config(scheduler_config)


def get_scheduler_class(scheduler_name):
    if scheduler_name.startswith("k_"):
        scheduler_name.replace("k_", "", 1)

    match scheduler_name:
        case "ddim":            return DDIMScheduler
        case "euler":           return EulerDiscreteScheduler
        case "euler_a":         return EulerAncestralDiscreteScheduler
        case "dpm_2":           return KDPM2DiscreteScheduler
        case "dpm_2_a":         return KDPM2AncestralDiscreteScheduler
        case "dpmpp_2m":        return DPMSolverMultistepScheduler
        case "dpmpp_2m_sde":    return DPMSolverMultistepScheduler
        case "dpmpp_sde":       return DPMSolverSinglestepScheduler
        case "heun":            return HeunDiscreteScheduler
        case "lms":             return LMSDiscreteScheduler
        case "pndm":            return PNDMScheduler
        case "unipc":           return UniPCMultistepScheduler
        case _:                 raise NotImplementedError


def get_scheduler_config(scheduler_name, current_scheduler_config):
    if scheduler_name.startswith("k_"):
        current_scheduler_config["use_karras_sigmas"] = True
    match scheduler_name:
        case "dpmpp_2m":
            current_scheduler_config["algorithm_type"] = "dpmsolver++"
            current_scheduler_config["solver_order"] = 2
        case "dpmpp_2m_sde":
            current_scheduler_config["algorithm_type"] = "sde-dpmsolver++"
    return current_scheduler_config


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Diffuser specific arguments
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--scheduler", type=str, default="dpmpp_2m", help="Scheduler name")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # DistriFuser specific arguments
    parser.add_argument("--no_split_batch", action="store_true", help="Disable the batch splitting for classifier-free guidance")
    parser.add_argument("--warmup_steps", type=int, default=4, help="Number of warmup steps")
    parser.add_argument(
        "--sync_mode",
        type=str,
        default="corrected_async_gn",
        choices=["separate_gn", "stale_gn", "corrected_async_gn", "sync_gn", "full_sync", "no_sync"],
        help="Different GroupNorm synchronization modes",
    )
    parser.add_argument(
        "--parallelism",
        type=str,
        default="patch",
        choices=["patch", "tensor", "naive_patch"],
        help="patch parallelism, tensor parallelism or naive patch",
    )
    parser.add_argument("--no_cuda_graph", action="store_true", help="Disable CUDA graph")
    parser.add_argument(
        "--split_scheme",
        type=str,
        default="alternate",
        choices=["row", "col", "alternate"],
        help="Split scheme for naive patch",
    )

    # Added arguments
    parser.add_argument("--model_path", type=str, default=None, help="Path to model folder")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--variant", type=str, default="fp16", help="PyTorch variant [fp16/fp32]")
    parser.add_argument("--pipeline_type", type=str, default="SDXL", help="Stable Diffusion pipeline type [SD/SDXL]")
    parser.add_argument("--compel", action="store_true", help="Enable Compel")
    # parser.add_argument("--lora", type=str, default=None, help="A dictionary of LoRAs to load, with their weights")
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true")
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--enable_slicing", action="store_true")
    args = parser.parse_args()
    return args


def setup_logger():
    global logger
    global local_rank
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)


@app.route("/initialize", methods=["GET"])
def check_initialize():
    global initialized
    if initialized:
        return jsonify({"status": "initialized"}), 200
    else:
        return jsonify({"status": "initializing"}), 202


def initialize():
    global pipe, engine_config, input_config, local_rank, initialized
    mp.set_start_method("spawn", force=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    setup_logger()
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    args = get_args()
    assert (args.height > 0 and args.width > 0), "Invalid image dimensions"
    distri_config = DistriConfig(
        height=args.height,
        width=args.width,
        do_classifier_free_guidance=args.guidance_scale > 1,
        split_batch=not args.no_split_batch,
        warmup_steps=args.warmup_steps,
        mode=args.sync_mode,
        use_cuda_graph=not args.no_cuda_graph,
        parallelism=args.parallelism,
        split_scheme=args.split_scheme,
    )

    assert args.model_path is not None, "No model specified"

    assert args.variant in ["fp16", "fp32"], "Unsupported variant"

    assert args.pipeline_type in ["SD", "SDXL"], "Unsupported pipeline"
    if args.pipeline_type == "SDXL":
        PipelineClass = DistriSDXLPipeline
    else:
        PipelineClass = DistriSDPipeline

    pipe = PipelineClass.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        distri_config=distri_config,
        torch_dtype=torch.float16 if args.variant == "fp16" else torch.float32,
        use_safetensors=True,
    )

    pipe.pipeline.scheduler = get_scheduler(args.scheduler, pipe.pipeline.scheduler.config)

    if args.enable_slicing:
        pipe.pipeline.enable_vae_slicing()
    if args.enable_tiling:
        pipe.pipeline.enable_vae_tiling()
    if args.enable_model_cpu_offload:
        pipe.pipeline.enable_model_cpu_offload()
    if args.enable_sequential_cpu_offload:
        pipe.pipeline.enable_sequential_cpu_offload()
    # pipe.pipeline.enable_xformers_memory_efficient_attention()

    pipe.set_progress_bar_config(disable=distri_config.rank != 0)

    # if args.lora:
    #     loras = json.loads(args.lora)
    #     i = 0
    #     adapters_name = []
    #     adapters_weights = []
    #     for adapter, weight in loras.items():
    #         pipe.pipeline.load_lora_weights(adapter, adapter_name=f"adapter_{i}")
    #         adapters_name.append(f"adapter_{i}")
    #         i += 1
    #         logger.info(f"Loaded LoRA: {adapter}")
    #         adapters_weights.append(weight)
    #         logger.info(f"Set LoRA weight: {weight}")
    #     pipe.pipeline.set_adapters(adapters_name, adapter_weights=adapters_weights)

    logger.info("Model initialization completed")
    initialized = True  # 设置初始化完成标志
    return


def generate_image_parallel(
    positive_prompt, negative_prompt, num_inference_steps, seed, cfg, clip_skip
):
    global pipe, local_rank, input_config
    logger.info(f"Starting image generation with prompt: {positive_prompt}")
    logger.info(f"Negative: {negative_prompt}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    args = get_args()

    positive_embeds = None
    positive_pooled_embeds = None
    negative_embeds = None
    negative_pooled_embeds = None
    if args.compel:
        compel = Compel(
            tokenizer=[pipe.pipeline.tokenizer, pipe.pipeline.tokenizer_2],
            text_encoder=[pipe.pipeline.text_encoder, pipe.pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        positive_embeds, positive_pooled_embeds = compel([positive_prompt])
        if len(negative_prompt) > 0:
            negative_embeds, negative_pooled_embeds = compel([negative_prompt])
    
    output = pipe(
        prompt=positive_prompt if positive_embeds is None else None,
        negative_prompt=negative_prompt if negative_embeds is None else None,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=cfg,
        clip_skip=clip_skip,
        prompt_embeds=positive_embeds,
        pooled_prompt_embeds=positive_pooled_embeds,
        negative_embeds=negative_embeds,
        negative_pooled_embeds=negative_pooled_embeds,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")

    if dist.get_rank() != 0:
        # serialize output object
        output_bytes = pickle.dumps(output)

        # send output to rank 0
        dist.send(torch.tensor(len(output_bytes), device=f"cuda:{local_rank}"), dst=0)
        dist.send(torch.ByteTensor(list(output_bytes)).to(f"cuda:{local_rank}"), dst=0)

        logger.info(f"Output sent to rank 0")

    if dist.get_rank() == 0:
        # recv from rank world_size - 1
        size = torch.tensor(0, device=f"cuda:{local_rank}")
        dist.recv(size, src=dist.get_world_size() - 1)
        output_bytes = torch.ByteTensor(size.item()).to(f"cuda:{local_rank}")
        dist.recv(output_bytes, src=dist.get_world_size() - 1)

        # deserialize output object
        output = pickle.loads(output_bytes.cpu().numpy().tobytes())

    return output, elapsed_time

@app.route("/generate", methods=["POST"])
def generate_image():
    logger.info("Received POST request for image generation")
    data = request.json
    prompt = data.get("prompt")
    negative_prompt = data.get("negative_prompt")
    num_inference_steps = data.get("num_inference_steps")
    seed = data.get("seed")
    cfg = data.get("cfg", 8.0)
    clip_skip = data.get("clip_skip", 0)

    logger.info(
        f"Request parameters: prompt='{prompt}', negative_prompt='{negative_prompt}', steps={num_inference_steps}, seed={seed}, cfg={cfg}, clip_skip={clip_skip}"
    )
    # Broadcast request parameters to all processes
    params = [prompt, negative_prompt, num_inference_steps, seed, cfg, clip_skip]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    output, elapsed_time = generate_image_parallel(*params)

    # Ensure output is not None before accessing its attributes
    if output and hasattr(output, "images") and output.images:
        pickled_image = pickle.dumps(output)
        output_base64 = base64.b64encode(pickled_image).decode('utf-8')
    else:
        output_base64 = ""
    image_path = ""

    response = {
        "message": "Image generated successfully",
        "elapsed_time": f"{elapsed_time:.2f} sec",
        "output": output_base64,
    }

    logger.info("Sending response")
    return jsonify(response)


def run_host():
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="0.0.0.0", port=6000)
    else:
        while True:
            params = [None] * 6 # len(params) of generate_image_parallel()
            logger.info(f"Rank {dist.get_rank()} waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                logger.info("Received exit signal, shutting down")
                break
            logger.info(f"Received task with parameters: {params}")
            generate_image_parallel(*params)


if __name__ == "__main__":
    initialize()

    logger.info(
        f"Process initialized. Rank: {dist.get_rank()}, Local Rank: {os.environ.get('LOCAL_RANK', 'Not Set')}"
    )
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    run_host()

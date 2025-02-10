#!/usr/bin/env python3
import argparse
import backend_pb2
import backend_pb2_grpc
import base64
import grpc
import json
import os
import pickle
import requests
import signal
import subprocess
import sys
import time
import torch
from concurrent import futures
from pathlib import Path
from PIL import Image


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
COMPEL = os.environ.get("COMPEL", "0") == "1"
WARMUP_STEPS = 0
HOST_INIT_TIMEOUT = 90
PORT = "6000"
MASTER_PORT = "29400"
URL = f"http://localhost:{PORT}"


# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))


process = None


def kill_process():
    global process
    if process is not None:
        process.terminate()
        time.sleep(15)
        process = None


# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    last_height = None
    last_width = None
    last_model_path = None
    last_cfg_scale = 7
    last_scheduler = None
    variant = "fp32"
    loras = {}
    last_clip_skip = 0
    is_low_vram = False

    is_loaded = False
    needs_reload = False


    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))


    def LoadModel(self, request, context):
        if request.CFGScale != 0 and self.last_cfg_scale != request.CFGScale:
            self.last_cfg_scale = request.CFGScale

        if request.Model != self.last_model_path or (
            len(request.ModelFile) > 0 and os.path.exists(request.ModelFile) and request.ModelFile != self.last_model_path
        ):
            self.needs_reload = True

        self.last_model_path = request.Model
        if request.ModelFile != "":
            if os.path.exists(request.ModelFile):
                self.last_model_path = request.ModelFile

        if request.F16Memory and self.variant != "fp16":
            self.needs_reload = True
            self.variant = "fp16"
        elif not request.F16Memory and self.variant != "fp32":
            self.needs_reload = True
            self.variant = "fp32"

        if request.SchedulerType != self.last_scheduler:
            self.needs_reload = True
            self.last_scheduler = request.SchedulerType

        if request.LoraAdapters:
            if len(self.loras.keys()) == 0 and len(request.LoraAdapters) == 0:
                pass
            elif len(self.loras.keys()) != len(request.LoraAdapters):
                self.needs_reload = True
            else:
                for adapter in self.loras.keys():
                    if adapter not in request.LoraAdapters:
                        self.needs_reload = True
                        break
            if self.needs_reload:
                self.loras = {}
                if len(request.LoraAdapters) > 0:
                    i = 0
                    for adapter in request.LoraAdapters:
                        self.loras[adapter] = request.LoraScales[i]
                        i += 1

        if request.CLIPSkip != self.last_clip_skip:
            self.last_clip_skip = request.CLIPSkip

        if self.is_low_vram != request.LowVRAM:
            self.needs_reload = True
            self.is_low_vram = request.LowVRAM

        return backend_pb2.Result(message="", success=True)


    def GenerateImage(self, request, context):
        if request.height != self.last_height or request.width != self.last_width:
            self.needs_reload = True

        if not self.is_loaded or self.needs_reload:
            kill_process()
            nproc_per_node = torch.cuda.device_count()
            # LoRAs will error loading with nproc_per_node > 2
            if len(self.loras) > 1 and nproc_per_node > 2:
                nproc_per_node = 2
            self.last_height = request.height
            self.last_width = request.width

            pipeline_type = "SD"
            sdxl_model_names_lower = ["sdxl", "_xl", "-xl", " xl"]
            for name in sdxl_model_names_lower:
                if name in self.last_model_path.lower():
                    pipeline_type = "SDXL"
                    break
            if "XL" in self.last_model_path:
                pipeline_type = "SDXL"

            cmd = [
                'torchrun',
                f'--nproc_per_node={nproc_per_node}',
                f'--master-port={MASTER_PORT}',
                'host.py',

                f'--port={PORT}',
                '--host_mode=localai',
                f'--model_path={self.last_model_path}',
                f'--pipeline_type={pipeline_type}',
                f'--variant={self.variant}',
                f'--height={self.last_height}',
                f'--width={self.last_width}',
                f'--no_cuda_graph',
                f'--warmup_steps={WARMUP_STEPS}',
                '--parallelism=patch',
            ]

            # enable for more vram usage, and slower
            # best to leave this disabled
            no_split_batch = False
            if no_split_batch:
                cmd.append('--no_split_batch')

            if COMPEL:
                cmd.append('--compel')

            if len(self.loras) > 0:
                cmd.append(f'--lora={json.dumps(self.loras)}')

            if self.last_scheduler is not None and len(self.last_scheduler) > 0:
                cmd.append(f'--scheduler={self.last_scheduler}')

            if self.is_low_vram:
                # cmd.append('--enable_model_cpu_offload')          # breaks parallelism
                # cmd.append('--enable_sequential_cpu_offload')     # crash
                cmd.append('--enable_tiling')
                cmd.append('--enable_slicing')
                cmd.append('--xformers_efficient')

            global process
            process = subprocess.Popen(cmd)
            initialize_url = f"{URL}/initialize"
            time_elapsed = 0
            while True:
                try:
                    response = requests.get(initialize_url)
                    if response.status_code == 200 and response.json().get("status") == "initialized":
                        self.is_loaded = True
                        self.needs_reload = False
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
                time_elapsed += 1
                if time_elapsed > HOST_INIT_TIMEOUT:
                    kill_process()
                    return backend_pb2.Result(message=f"Failed to launch host within {HOST_INIT_TIMEOUT} seconds", success=False)

        if self.is_loaded:
            url = f"{URL}/generate"
            data = {
                "positive_prompt": request.positive_prompt,
                "num_inference_steps": request.step,
                "seed": request.seed,
                "cfg": self.last_cfg_scale,
                "clip_skip": self.last_clip_skip,
            }

            if request.negative_prompt and len(request.negative_prompt) > 0:
                data["negative_prompt"] = request.negative_prompt

            response = requests.post(url, json=data)
            response_data = response.json()
            output_base64 = response_data.get("output", "")

            if output_base64:
                output_bytes = base64.b64decode(output_base64)
                output = pickle.loads(output_bytes)
            else:
                output = None
                kill_process()
                assert False, "No output object received"
            image = output.images[0]
            if image.size == (0,0) or not image.getbbox():
                return backend_pb2.Result(message="No image generated", success=False)
            else:
                image.save(request.dst)
                return backend_pb2.Result(message="Media generated", success=True)
        else:
            return backend_pb2.Result(message="Host is not loaded", success=False)


def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)

    # Define the signal handler function
    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...")
        kill_process()
        server.stop(0)
        sys.exit(0)

    # Set the signal handlers for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        kill_process()
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument("--addr", default="localhost:50051", help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)

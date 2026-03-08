import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import argparse
import math
from conv_for_infer import generate_conv
from inference_runtime import (
    batched,
    build_worker_assignments,
    configure_single_gpu_process,
)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--processor_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    return parser.parse_args()


def collect_images(input_dir):
    images = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, name))
    return images

def worker(
    rank,
    gpu_id,
    image_paths,
    model_dir,
    processor_dir,
    output_dir,
    batch_size,
    max_model_len,
    max_num_seqs,
    gpu_memory_utilization,
):
    configure_single_gpu_process(gpu_id)

    from dataclasses import asdict
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor
    from vllm import LLM, EngineArgs, SamplingParams

    print(f"[Worker {rank}] Using GPU {gpu_id}, images: {len(image_paths)}")

    processor = AutoProcessor.from_pretrained(processor_dir)

    engine_args = EngineArgs(
        model=model_dir,
        max_model_len=max_model_len,
        max_num_seqs=max(batch_size, max_num_seqs),
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=gpu_memory_utilization,
    )
    engine_args = asdict(engine_args) | {"seed": 0}

    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
    )

    num_batches = math.ceil(len(image_paths) / batch_size)
    for image_batch in tqdm(batched(image_paths, batch_size), total=num_batches, desc=f"GPU {gpu_id}"):
        requests = []
        basenames = []

        for image_path in image_batch:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            basenames.append(basename)
            data_dict = {
                "image_path": image_path
            }
            messages = generate_conv(data_dict)
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            with Image.open(image_path) as image:
                requests.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": [image.copy()]},
                    }
                )

        outputs = llm.generate(requests, sampling_params=sampling_params)

        for basename, output in zip(basenames, outputs):
            markdown_file = os.path.join(output_dir, f"{basename}.md")
            text = output.outputs[0].text
            with open(markdown_file, "w", encoding="utf-8") as f:
                f.write(text)


def main():
    args = parse_args()

    import torch

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = collect_images(args.input_dir)
    assert len(image_paths) > 0, "No images found"

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA devices found"

    print(f"Detected {num_gpus} GPUs, total images: {len(image_paths)}")

    assignments = build_worker_assignments(image_paths, num_gpus)

    processes = []
    for rank, (gpu_id, chunk) in enumerate(assignments):
        p = mp.Process(
            target=worker,
            args=(
                rank,
                gpu_id,
                chunk,
                args.model_dir,
                args.processor_dir,
                args.output_dir,
                args.batch_size,
                args.max_model_len,
                args.max_num_seqs,
                args.gpu_memory_utilization,
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

"""
python qwen3_vllm_infer.py \
    --model_dir /workspace/Qwen3-VL-2B-Instruct \
    --processor_dir /workspace/Qwen3-VL-2B-Instruct \
    --input_dir /workspace/cropped \
    --output_dir /workspace/outputs
"""

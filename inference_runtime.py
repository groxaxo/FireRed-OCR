import os


def configure_single_gpu_process(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def split_list(data, n):
    if n <= 0:
        raise ValueError("n must be greater than 0")
    return [data[i::n] for i in range(n)]


def build_worker_assignments(image_paths, num_gpus):
    return [
        (gpu_id, chunk)
        for gpu_id, chunk in enumerate(split_list(image_paths, num_gpus))
        if chunk
    ]


def batched(data, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    for index in range(0, len(data), batch_size):
        yield data[index:index + batch_size]


def get_hf_model_load_kwargs():
    return {
        "torch_dtype": "auto",
        "device_map": {"": "cuda:0"},
    }

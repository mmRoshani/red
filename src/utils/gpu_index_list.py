from torch import cuda

def list_available_gpus():
    available_gpus = []
    if cuda.is_available():
        num_gpus = cuda.device_count()
        for i in range(num_gpus):
            gpu_name = cuda.get_device_name(i)
            available_gpus.append((i, gpu_name))
    return available_gpus
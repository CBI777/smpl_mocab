import torch

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # Convert bytes to GB
        cached_memory = torch.cuda.memory_cached() / (1024**3)  # Convert bytes to GB
        return allocated_memory, cached_memory
    else:
        return 0.0, 0.0

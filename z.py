import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch can use the GPU.")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available. PyTorch cannot use the GPU.")

if __name__ == "__main__":
    check_cuda()
import torch


def dialate_tensor(tensor, kernel_size=3):
    """
    Dialates a tensor by a given kernel size.
    args:
        tensor: tensor to dialate (C, H, W)
        kernel_size: size of dialation kernel
    """
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32).to(tensor.device)
    return torch.nn.functional.conv2d(
        tensor.unsqueeze(1), kernel, padding=kernel_size // 2
    ).squeeze(1)
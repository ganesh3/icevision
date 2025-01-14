__all__ = [
    "to_np",
    "tensor_to_image",
    "requires_grad",
    "model_device",
    "params",
    "check_all_model_params_in_groups2",
    "model_device",
    "auto_device_config",
]

from icevision.imports import *


def to_np(t):
    return t.detach().cpu().numpy()


def tensor_to_image(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().transpose(1, 2, 0)


def requires_grad(model, layer):
    return list(model.parameters())[layer].requires_grad


def model_device(model):
    return first(model.parameters()).device


def auto_device_config(device: None):

    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else device
    )


def params(m):
    return list(m.parameters())


def check_all_model_params_in_groups2(
    model: nn.Module, param_groups: List[List[nn.Parameter]]
):
    num_params = len([param for group in param_groups for param in group])
    num_params_expected = len(list(model.parameters()))

    if num_params != num_params_expected:
        raise RuntimeError(
            f"{num_params_expected} params in model but only {num_params} "
            "in parameter group"
        )


def model_device(model: nn.Module):
    """Returns the device the first model parameter is stored.

    Can be wrong if different parts of the model are in different devices.
    """
    return next(iter(model.parameters())).device

import torch
from tqdm import tqdm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def parameters_to_vector(parameters):
    """
    Same as https://pytorch.org/docs/stable/generated/torch.nn.utils.parameters_to_vector.html
    but with :code:`reshape` instead of :code:`view` to avoid a pesky error.
    """
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return torch.cat(vec)


def grad_calculator(data_loader,
                    model,
                    parameters,
                    func,
                    normalize_factor,
                    device,
                    projector,
                    checkpoint_id):
    all_grads = []
    for _, data in enumerate(data_loader):
        model_output = func(data, model)
        if torch.isinf(model_output):
            # TODO: handle numerical problem better
            # print("numerical problem happens, model output function equals to inf")
            grads = torch.zeros(count_parameters(model), dtype=torch.float32).to(device)
            if projector is None:
                grads_p = grads.clone().detach().unsqueeze(0)
            else:
                grads_p = projector.project(grads.clone().detach().unsqueeze(0), model_id=checkpoint_id, is_grads_dict=False)
            all_grads.append(grads_p)
        else:
            grads = parameters_to_vector(torch.autograd.grad(model_output, parameters, retain_graph=True)).to(device)
            grads /= normalize_factor
            if projector is None:
                grads_p = grads.clone().detach().unsqueeze(0)
            else:
                grads_p = projector.project(grads.clone().detach().unsqueeze(0), model_id=checkpoint_id, is_grads_dict=False)
            all_grads.append(grads_p)
    all_grads = torch.cat(all_grads, dim=0)
    return all_grads


def out_to_loss_grad_calculator(data_loader,
                                model,
                                func):
    out_to_loss_grads = []
    for _, data in enumerate(tqdm(data_loader)):
        out_to_loss_grad = func(data, model)
        out_to_loss_grads.append(out_to_loss_grad)
    return torch.diag(torch.cat(out_to_loss_grads).reshape(-1))

import torch
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from src.nets.network_factory import network_factory
from src.utils.similarities.pairwise_coordinate_similarity import cosine_similarity
from src.utils.similarities.stubs import vectorise_model
from src.utils.log import Log
from src.validators.config_validator import ConfigValidator


def global_prune_without_masks(model, amount):
    """Global Unstructured Pruning of model."""
    parameters_to_prune = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            if isinstance(mod.weight, torch.nn.Parameter):
                parameters_to_prune.append((mod, "weight"))
        if hasattr(mod, "bias"):
            if isinstance(mod.bias, torch.nn.Parameter):
                parameters_to_prune.append((mod, "bias"))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for mod in model.modules():
        if hasattr(mod, "weight_orig"):
            if isinstance(mod.weight_orig, torch.nn.Parameter):
                prune.remove(mod, "weight")
        if hasattr(mod, "bias_orig"):
            if isinstance(mod.bias_orig, torch.nn.Parameter):
                prune.remove(mod, "bias")


def calculate_optimal_sensitivity_percentage(example_client_model, config: ConfigValidator, log: Log):
    prune_rate = torch.linspace(0, 1, 101)
    cosine_sim = []
    base_vec = vectorise_model(example_client_model)
    prune_net = network_factory(
        model_type=config.MODEL_TYPE,
        number_of_classes=config.NUMBER_OF_CLASSES,
        pretrained=config.PRETRAINED_MODELS,
        model_size= config.TRANSFORMER_MODEL_SIZE,
    )
    prune_net.to(config.DEVICE)

    log.info("starting calculating optimal sensitivity percentage...")

    for p in prune_rate:
        p = float(p)
        prune_net.load_state_dict(example_client_model.state_dict())
        global_prune_without_masks(prune_net, p)
        prune_net_vec = vectorise_model(prune_net)
        cosine_sim.append(cosine_similarity(base_vec, prune_net_vec).item())

    c = torch.vstack((torch.Tensor(cosine_sim), prune_rate))
    d = c.T
    dists = []
    for i in d:
        dists.append(torch.dist(i, torch.Tensor([1, 1])))
    min = torch.argmin(torch.Tensor(dists))

    del dists

    _plot_name =  f'{config.MODEL_TYPE} Parateo Front'
    plt.plot(
        prune_rate, cosine_sim, label=_plot_name
    )
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.scatter(1, 1, label="Utopia", c="red", marker="*", s=150)
    plt.scatter(prune_rate[min], cosine_sim[min], color="k", marker="o", label="Optima")
    plt.xlabel(xlabel="pruning rate")
    plt.ylabel(ylabel="cosine similarity")
    plt.legend()
    plt.grid()
    plt.savefig(f'{_plot_name}.png')
    # plt.show()

    del cosine_sim
    del base_vec
    del prune_net

    optimal_sensitivity_percentage = (1.0 - prune_rate[min]) * 100
    del prune_rate

    return optimal_sensitivity_percentage
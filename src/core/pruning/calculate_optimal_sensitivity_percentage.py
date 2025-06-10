import torch
import os
from src.constants import PLOT_PATH
from src.core.pruning.clip_cosine_similarity import clip_cosine_similarity
from src.core.pruning.global_prune_without_masks import global_prune_without_masks
from src.utils.vectorise_model_parameters import vectorise_model_parameters
from src.utils.log import Log
from src.validators.config_validator import ConfigValidator
import matplotlib.pyplot as plt


def calculate_optimal_sensitivity_percentage(
        example_client_model,
        prune_net,
        config: 'ConfigValidator',
        log: 'Log',
        plot_save_path = f'./{PLOT_PATH}'):

    prune_rate = torch.linspace(0, 1, 101) # RvQ: linespace?
    cosine_sim = [] # RvQ: why vector?
    base_vec = vectorise_model_parameters(example_client_model)

    log.info("starting calculating optimal sensitivity percentage...")

    for p in prune_rate:
        p = float(p)
        prune_net.load_state_dict(example_client_model.state_dict())
        global_prune_without_masks(prune_net, p)
        prune_net_vec = vectorise_model_parameters(prune_net)
        cosine_sim.append(clip_cosine_similarity(base_vec, prune_net_vec).item())

    c = torch.vstack((torch.Tensor(cosine_sim), prune_rate))
    d = c.T
    dists = []
    for i in d:
        dists.append(torch.dist(i, torch.Tensor([1, 1])))
    min = torch.argmin(torch.Tensor(dists))

    del dists

    # RvQ: the fuck we be plotting here?
    plt.plot(
        prune_rate, cosine_sim, label=f'{config.MODEL_TYPE} Parateo Front'
    )
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.scatter(1, 1, label="Utopia", c="red", marker="*", s=150)
    plt.scatter(prune_rate[min], cosine_sim[min], color="k", marker="o", label="Optima")
    plt.xlabel(xlabel="pruning rate")
    plt.ylabel(ylabel="cosine similarity")
    plt.legend()
    plt.grid()

    saving_path: str = f"{plot_save_path}/{config.MODEL_TYPE}/{config.DISTANCE_METRIC}"
    os.makedirs(saving_path, exist_ok=True)

    plt.savefig(f"{saving_path}/optimal_pruning.png")

    del cosine_sim
    del base_vec
    del prune_net

    optimal_sensitivity_percentage = (1.0 - prune_rate[min]) * 100
    del prune_rate

    return optimal_sensitivity_percentage
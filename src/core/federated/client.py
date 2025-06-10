# from core.federated.federated_base import FederatedTrainingDevice
# from utils.log import Log
# from validators.config_validator import ConfigValidator
# import copy as py_copy
# from core.federated.train import train
# import torch
# import torch.nn.functional as F
# from torch import autograd
#
# class Client(FederatedTrainingDevice):
#     def __init__(self, model, optimizer_fn, id_num, train_data_loader, evaluation_data_loader, config: 'ConfigValidator', log: 'Log'):
#         super().__init__(model, config, log)
#
#         self.optimizer = optimizer_fn(self.model.parameters())
#
#         self.train_loader = train_data_loader
#         self.eval_loader = evaluation_data_loader
#
#         self.gradients = {}
#
#         self.id = id_num
#
#         self.log.info(f"client no: {self.id} initialized")
#
#     def synchronize_with_server(self, server):
#         self.model.load_state_dict(server.model.state_dict())
#
#     def compute_weight_update(
#             self,
#             be_ready_for_clustering: bool,
#             epochs: int,
#             loader=None,
#     ):
#         # ---------- 1. local training --------------------------------------------------
#         _updated_model, train_stats = train(
#             self.model,
#             self.train_loader if loader is None else loader,
#             self.optimizer,
#             epochs,
#         )
#         self.model.load_state_dict(_updated_model.state_dict())
#         del _updated_model
#
#         self.log.info(f"training done for client no {self.id} with loss of {train_stats}")
#
#         # ---------- 2. Fisher-diagonal computation (optional) --------------------------
#         if not be_ready_for_clustering:
#             return train_stats
#
#         # Copy so the main model stays untouched
#         _model = py_copy.deepcopy(self.model).to(self.device)
#         _model.eval()
#
#         # Allocate one tensor per parameter (same shape, same device)
#         fisher_diag = [torch.zeros_like(p, device=self.device) for p in _model.parameters()]
#
#         for inputs, targets in self.train_loader:
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#
#             # log-probabilities for the current batch
#             log_probs = F.log_softmax(_model(inputs), dim=1)
#
#             # ---- per-sample score function -------------------------------------------
#             for i, tgt in enumerate(targets):
#                 # scalar log-probability of the correct class
#                 log_p_correct = log_probs[i, tgt]
#
#                 # gradient of log p(y|x;θ)  w.r.t. θ  (first-order score)
#                 _model.zero_grad(set_to_none=True)
#                 grads = autograd.grad(
#                     log_p_correct,
#                     _model.parameters(),
#                     create_graph=False,  # we only need first-order grads
#                     retain_graph=False,
#                 )
#                 # accumulate squared gradients
#                 for fd, g in zip(fisher_diag, grads):
#                     fd.add_(g.detach() ** 2)
#
#         # ---- 3. expectation over the dataset -----------------------------------------
#         num_samples = len(self.train_loader.dataset)
#         fisher_diag = [fd / num_samples for fd in fisher_diag]
#
#         # ---- 4. layer-wise min-max normalisation -------------------------------------
#         eps = 1e-12  # avoids division by zero
#         norm_fisher_diag = []
#         for fd in fisher_diag:
#             x_min, x_max = fd.min(), fd.max()
#             norm_fisher_diag.append((fd - x_min) / (x_max - x_min + eps))
#
#         # keep a flat copy that matches your previous self.gradients structure
#         flat = torch.cat([fd.view(-1).cpu() for fd in norm_fisher_diag]).numpy()
#         self.gradients = {i: v for i, v in enumerate(flat)}
#         self.log.info(f"Fisher diagonal computed with {len(self.gradients)} entries.")
#
#         del _model, fisher_diag, norm_fisher_diag  # free memory
#         torch.cuda.empty_cache()
#
#         return train_stats
from learning.evaluation import model_evaluation
from utils.checker import device_checker
from utils.log import Log
from typing import Any, Dict

from validators.config_validator import ConfigValidator


class FederatedBase(object):

        def __init__(
                self,
                nodes: List[VirtualNode],
                topology: Union[str, np.ndarray],
                resources: Union[str, PlacementGroup] = "uniform",
                federation_id: str = "",
                is_tune: bool = False,
                bundle_offset: int = 0,
        ):
    def __init__(
        self,
        model,
        config: 'ConfigValidator', log: 'Log'):
        self.log = log
        self.config = config
        self.device = device_checker(self.config.DEVICE)
        self.model = model.to(self.device)

    def evaluate(self):
        _loss, _accuracy = model_evaluation(self.model, self.eval_loader)

        if _loss < 1.0 and _accuracy > 0.6:
            self.log.info(
                f"testing done for client no {self.id} with accuracy of {_accuracy} and loss of {_loss} [GOOD]"
            )
        elif _loss < 2.0 and _accuracy > 0.4:
            self.log.warn(
                f"testing done for client no {self.id} with accuracy of {_accuracy} and loss of {_loss} [MODERATE]"
            )
        else:
            self.log.warn(
                f"testing done for client no {self.id} with accuracy of {_accuracy} and loss of {_loss} [POOR]"
            )

        return _accuracy
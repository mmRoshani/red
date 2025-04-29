from learning.evaluation import model_evaluation
from utils.checker import device_checker

class FederatedTrainingDevice(object):
    def __init__(self, model, log, device: str):

        self.log = log
        self.device = device_checker(device)
        self.model = model.to(device)

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
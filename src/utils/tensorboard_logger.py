from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


class TensorBoardLogger:
    def __init__(self, log_dir="runs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)

    def log_metrics(self, metrics_dict, step):
        """Log multiple metrics at once"""
        for metric_name, metric_value in metrics_dict.items():
            self.writer.add_scalar(metric_name, metric_value, step)

    def log_model_parameters(self, model, step):
        """Log model parameters and gradients"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"parameters/{name}", param.data, step)
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"gradients/{name}", param.grad.data, step
                    )

    def close(self):
        """Close the writer"""
        self.writer.close()

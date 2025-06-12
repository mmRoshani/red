<<<<<<< HEAD:src/utils/count_parameters.py
from prettytable import PrettyTable
from pprint import pformat
from utils.log import Log

def count_parameters(model, model_type: str, log: Log):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    log.info(f"Total trainable parameter for model {model_type} is:\n{total_params}")
    log.info("\n" + pformat(table))

=======
from prettytable import PrettyTable
from pprint import pformat
from utils.log import Log

def count_parameters(model, model_type: str, log: Log):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    log.info(f"Total trainable parameter for model {model_type} is:\n{total_params}")
    log.info("\n" + pformat(table))

>>>>>>> dev:utils/count_parameters.py
    return total_params
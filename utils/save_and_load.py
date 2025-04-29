from torch import load, save


def load_torch_model(node_id):
    model_path = f"models/node_{node_id}.pth"
    model = load(model_path)
    return model


def load_torch_model_before_agg(node_id):
    model_path = f"models/before_aggregation/node_{node_id}.pth"
    model = load(model_path)
    return model


def save_torch_model_before_agg(model, client_id: str):
    model_path = f"models/before_aggregation/node_{client_id}.pth"
    save(model, model_path)


def save_torch_model(model, node_id):
    model_path = f"models/node_{node_id}.pth"
    save(model, model_path)


def save_model_param(model, node_id, round_number):
    model_path = f"models/node_{node_id}_round_{round_number}.pth"
    save(model.state_dict(), model_path)

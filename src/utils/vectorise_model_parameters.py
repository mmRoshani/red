from torch.nn.utils import parameters_to_vector

def vectorise_model_parameters(model):
    return parameters_to_vector(model.parameters())

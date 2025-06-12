from torch.nn.utils import parameters_to_vector as Params2Vec

def vectorise_model(model):
    return Params2Vec(model.parameters())
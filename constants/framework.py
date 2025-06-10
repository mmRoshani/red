# PATHS
LOG_PATH: str = "./logs"
MODELS_SAVING_PATH: str = "./models"
GLOBAL_MODELS_SAVING_PATH: str = f"{MODELS_SAVING_PATH}/after_aggregation"
PLOT_PATH: str = './plots'
DATA_PATH: str = "~/data"

# RESOURCES
TOPOLOGY_MANAGER_CPU_RESOURCES: float | int = 0.5 # RvQ: TF?
SAFETY_EPSILON: float= 0.01 # RvQ: TF???


# Algorithm
SERVER_ID: str = "server"

# Communication
MODEL_UPDATE: str = "model_update"
SIMMILARITY_REQUEST : str = "similarity_request"

MESSAGE_BODY_STATES = "state"
MESSAGE_BODY_NORM = "norm"
SIMMILARITY_REQUEST_ACCEPT = "simmilarity_request_accept"
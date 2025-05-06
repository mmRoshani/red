from utils.log import Log

def client_ids_list_generator(n_clients_or_ids, log: Log):

    clients_list = [f"client_{i}" for i in range(n_clients_or_ids)]

    log.info(f'generated a list for client_ids in config.RUNTIME {clients_list}')

    return clients_list
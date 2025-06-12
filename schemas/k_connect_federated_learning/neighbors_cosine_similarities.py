from constants.framework import SIMILARITY_REQUEST, SIMILARITY_REQUEST_APROVE
import torch

class NeighborsCosineSimilarities():

    def request(sender):
        
        model_vector = torch.nn.utils.parameters_to_vector(sender.model.parameters())

        message_body = {
            'sender_id': sender.id,
            'round': sender.local_round_counter,
            'model_vector': model_vector,
        }

        sender.log.info(f"Client {sender.id} sending model vertor to {len(sender.neighbors)} neighbors: {sender.neighbors}")
        
        for neighbor_id in sender.neighbors:
            sender.send(header=SIMILARITY_REQUEST, body=message_body, to=neighbor_id)
        return
    
    def request_aprove(receiver, sender_model_vector , sender_id):

        receiver_model_vector = torch.nn.utils.parameters_to_vector(receiver.model.parameters())
        sender_model_norm = torch.norm(sender_model_vector)
        receiver_model_norm = torch.norm(receiver_model_vector)

        cosine_similarity = torch.dot(receiver_model_vector,sender_model_vector)/(receiver_model_norm*sender_model_norm)

        message_body = {
            'sender_id': receiver.id,
            'round': receiver.local_round_counter,
            'cosine_similarity': cosine_similarity,
        }

        receiver.log.info(f"Client {receiver.id} sending cosine similarity to {sender_id}")
        
        receiver.send(header=SIMILARITY_REQUEST_APROVE, body=message_body, to=sender_id)

        return(cosine_similarity)
    
    def receive_request(receiver):
        receiver.neighbors_models = {}
        received_count = 0
        expected_neighbors = len(receiver.neighbors)
        
        receiver.log.info(f'Client {receiver.id} waiting to receive models from {expected_neighbors} neighbors')
        
        while received_count < expected_neighbors:
            message = receiver.receive(block=True, timeout=300.0)
            
            if message is None:
                receiver.log.warn(f'Client {receiver.id} timed out waiting for neighbor models')
                continue

            #Khamideh updates : similarity requests-----------------------
            elif message.header == SIMILARITY_REQUEST:
                sender_id = message.body.get('sender_id')
                sender_model_vector = message.body.get('model_vector')

                if 1 == 1 : # for future attack detect!
                    similarity = NeighborsCosineSimilarities.request_aprove(receiver,sender_model_vector,sender_id).item()
                    similarity_with_sender = {f"{sender_id}, local round {message.body.get("round")}" : f"round:{receiver.local_round_counter} similarity:{similarity}"}
                    receiver.similarity_dict.update(similarity_with_sender)

            elif message.header == SIMILARITY_REQUEST_APROVE:

                sender_id = message.body.get('sender_id')
                sender_model_vector = message.body.get('model_vector')
                similarity = (message.body.get("cosine_similarity")).item()

                if 1 == 1 : # for future attack detect!
                    similarity_with_sender = {f"{sender_id}, local round {message.body.get("round")}" : f"round:{receiver.local_round_counter} similarity:{similarity}"}
                    receiver.similarity_dict.update(similarity_with_sender)
                    receiver.log.info(f"Client {receiver.id} similarities : {receiver.similarity_dict}")

                    received_count += 1
            else:
                receiver.log.warn(f'Client {receiver.id} received unexpected message: {message.header}')
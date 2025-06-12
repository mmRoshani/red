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
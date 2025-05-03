# Message

A message constitutes a fundamental data structure employed for inter-node communication within a federated learning system.
It comprises a header, the identifier of the sending node, a timestamp indicating its creation, and a body.
The message body is implemented as a dictionary, enabling the transmission of arbitrary data necessary for coordination and information exchange between participating nodes.
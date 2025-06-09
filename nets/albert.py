import torch.nn as nn
from transformers import AlbertModel, AlbertConfig

class Albert(nn.Module): 
    def __init__(self, number_of_classes=None, pretrained_model=True):
        super(Albert, self).__init__()
        if pretrained_model:
            self.albert = AlbertModel.from_pretrained("albert-base-v2")
        else:
            config = AlbertConfig()
            self.albert = AlbertModel(config)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.albert.config.hidden_size, number_of_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

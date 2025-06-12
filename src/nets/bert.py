import torch.nn as nn
from transformers import BertModel, BertConfig
from src.constants.models_constants import TRANSFORMER_MODEL_SIZE_BASE, TRANSFORMER_MODEL_SIZE_LARGE

class Bert(nn.Module):
    def __init__(self, number_of_classes: int=None, pretrained_model: bool=True, model_size: str =TRANSFORMER_MODEL_SIZE_BASE):
        super(Bert, self).__init__()
        self._model_variation: str = f"bert-{model_size}-uncased"
        if (pretrained_model):
            self.bert = BertModel.from_pretrained(self._model_variation)
        else :
            _config = BertConfig()
            self.bert = BertModel(_config)
        
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, number_of_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

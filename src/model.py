import torch
import torch.nn as nn
from transformers import BertModel

class DepBERT(nn.Module):
    """
    DepBERT model architecture.
    Combines BERT with a CNN for processing dependency features.
    """
    def __init__(self, n_classes, bert_model_name='bert-base-uncased'):
        super(DepBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # CNN for dependency features
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # The input size to the fully connected layer will depend on the max_len
        # For max_len=128, after one pool layer, it becomes 64x64.
        # 16 * 64 * 64 = 65536
        self.fc_dep = nn.Linear(16 * 64 * 64, 128)

        # Classifier
        self.classifier = nn.Linear(self.bert.config.hidden_size + 128, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, dep_features):
        # BERT forward pass
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = bert_output.pooler_output

        # CNN for dependency features
        # dep_features shape: [batch_size, 2, max_len, max_len]
        dep_output = self.pool(self.relu(self.conv1(dep_features)))
        dep_output = dep_output.view(dep_output.size(0), -1) # Flatten
        dep_output = self.relu(self.fc_dep(dep_output))

        # Concatenate BERT output and dependency features
        combined_output = torch.cat((pooled_output, dep_output), dim=1)

        # Classification
        logits = self.classifier(combined_output)
        return self.softmax(logits)

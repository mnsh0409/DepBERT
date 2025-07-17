import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from .preprocess import Preprocessor
from .model import DepBERT
import pandas as pd
import numpy as np

# --- Configuration ---
DATA_PATH = "data/sample_data.csv"
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 4
EPOCHS = 3
N_CLASSES = 7 # TGR, RGH, RGC, SOL, CFM, IRC, IRH
LEARNING_RATE = 2e-5

# --- Custom Dataset ---
class DialogueDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.label_map = {label: i for i, label in enumerate(np.unique(labels))}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.preprocessor.encode_text(text)
        dep_co_occurrence, dep_label = self.preprocessor.get_dependency_features(text)
        
        dep_features = np.stack([dep_co_occurrence, dep_label])

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'dep_features': torch.tensor(dep_features, dtype=torch.float32),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long)
        }

# --- Training and Evaluation Functions ---
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        dep_features = d["dep_features"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dep_features=dep_features
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            dep_features = d["dep_features"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dep_features=dep_features
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, np.mean(losses), f1

# --- Main Execution ---
if __name__ == "__main__":
    preprocessor = Preprocessor(model_name=MODEL_NAME, max_len=MAX_LEN)
    df = preprocessor.load_data(DATA_PATH)

    # For demonstration, we'll use the same data for train and test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = DialogueDataset(
        texts=df_train.text.to_numpy(),
        labels=df_train.label.to_numpy(),
        preprocessor=preprocessor
    )
    
    test_dataset = DialogueDataset(
        texts=df_test.text.to_numpy(),
        labels=df_test.label.to_numpy(),
        preprocessor=preprocessor
    )

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepBERT(n_classes=N_CLASSES).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss, val_f1 = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device
        )
        print(f'Val loss {val_loss} accuracy {val_acc} F1-score {val_f1}')
        print()

    print("Training complete.")

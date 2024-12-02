from transformers import AutoModel, AutoTokenizer, RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AdamW
from tqdm import tqdm
import argparse
import gc

def initialize_device():
    torch.cuda.empty_cache()
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_dataset(file_path, split_column, split_value):
    data_frame = pd.read_csv(file_path)
    filtered_data = data_frame[data_frame[split_column] == split_value]
    return filtered_data

def split_data(dataset, text_column, label_column, additional_text_column, test_ratio=0.2):
    text_train, text_val, label_train, label_val = train_test_split(
        dataset[text_column].tolist(),
        dataset[label_column].tolist(),
        test_size=test_ratio
    )
    additional_train, additional_val, _, _ = train_test_split(
        dataset[additional_text_column].tolist(),
        dataset[label_column].tolist(),
        test_size=test_ratio
    )
    return text_train, text_val, label_train, label_val, additional_train, additional_val

class DualInputDataset(Dataset):
    def __init__(self, primary_texts, primary_labels, auxiliary_texts, primary_tokenizer, auxiliary_tokenizer, max_length):
        self.primary_texts = primary_texts
        self.primary_labels = primary_labels
        self.auxiliary_texts = auxiliary_texts
        self.primary_tokenizer = primary_tokenizer
        self.auxiliary_tokenizer = auxiliary_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.primary_texts)

    def __getitem__(self, index):
        primary_encoded = self.primary_tokenizer(
            self.primary_texts[index], 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )
        auxiliary_encoded = self.auxiliary_tokenizer(
            self.auxiliary_texts[index], 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )
        return (
            primary_encoded['input_ids'].squeeze(),
            primary_encoded['attention_mask'].squeeze(),
            auxiliary_encoded['input_ids'].squeeze(),
            auxiliary_encoded['attention_mask'].squeeze(),
            self.primary_labels[index]
        )

class LinearProjectionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 2)
        )

    def forward(self, input_tensor):
        return self.layers(input_tensor)

class CombinedEmbeddingModel(nn.Module):
    def __init__(self, main_model, auxiliary_model, projection_layer, freeze_auxiliary=True):
        super().__init__()
        self.main_model = main_model
        self.auxiliary_model = auxiliary_model
        self.projection_layer = projection_layer

        if freeze_auxiliary:
            for parameter in self.auxiliary_model.parameters():
                parameter.requires_grad = False

    def forward(self, main_ids, main_masks, auxiliary_ids, auxiliary_masks):
        main_output = self.main_model(input_ids=main_ids, attention_mask=main_masks)
        main_embeddings = main_output.last_hidden_state[:, 0, :]
        main_embeddings = nn.LayerNorm(main_embeddings.size()[1:])(main_embeddings)

        auxiliary_output = self.auxiliary_model(input_ids=auxiliary_ids, attention_mask=auxiliary_masks)
        auxiliary_embeddings = auxiliary_output.last_hidden_state[:, 0, :]
        auxiliary_embeddings = nn.LayerNorm(auxiliary_embeddings.size()[1:])(auxiliary_embeddings)

        combined_embeddings = torch.cat((main_embeddings, auxiliary_embeddings), dim=1)
        return self.projection_layer(combined_embeddings)

def train_epoch(model, optimizer, criterion, data_loader, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for main_ids, main_masks, aux_ids, aux_masks, labels in tqdm(data_loader, desc='Training', dynamic_ncols=True):
        main_ids, main_masks, aux_ids, aux_masks, labels = (
            main_ids.to(device), 
            main_masks.to(device), 
            aux_ids.to(device), 
            aux_masks.to(device), 
            labels.to(device)
        )

        optimizer.zero_grad()
        outputs = model(main_ids, main_masks, aux_ids, aux_masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += len(labels)

    return total_loss / len(data_loader), correct_predictions / total_samples

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions, ground_truths = [], []

    with torch.no_grad():
        for main_ids, main_masks, aux_ids, aux_masks, labels in tqdm(data_loader, desc='Evaluating', dynamic_ncols=True):
            main_ids, main_masks, aux_ids, aux_masks, labels = (
                main_ids.to(device), 
                main_masks.to(device), 
                aux_ids.to(device), 
                aux_masks.to(device), 
                labels.to(device)
            )

            outputs = model(main_ids, main_masks, aux_ids, aux_masks)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    return accuracy_score(ground_truths, predictions)

def main(arguments):
    torch.manual_seed(arguments.random_seed)
    computation_device = initialize_device()

    dataset_paths = {
        "ETHOS": '/scratch/dataset_path_Ethos_Dataset_Binary',
    
    }

    dataset_file = dataset_paths[arguments.data_source]
    training_data = load_dataset(dataset_file, 'split_type', 'train')
    testing_data = load_dataset(dataset_file, 'split_type', 'test')

    main_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    auxiliary_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_primary, val_primary, train_labels, val_labels, train_auxiliary, val_auxiliary = split_data(
        training_data, 'text', 'label', 'auxiliary_column'
    )

    train_loader = DataLoader(
        DualInputDataset(train_primary, train_labels, train_auxiliary, main_tokenizer, auxiliary_tokenizer, max_length=512),
        batch_size=8, shuffle=True
    )
    validation_loader = DataLoader(
        DualInputDataset(val_primary, val_labels, val_auxiliary, main_tokenizer, auxiliary_tokenizer, max_length=512),
        batch_size=8, shuffle=False
    )

    model_main = AutoModel.from_pretrained("distilbert-base-uncased").to(computation_device)
    model_auxiliary = RobertaModel.from_pretrained("roberta-base").to(computation_device)
    projection_layer = LinearProjectionModel(input_dim=1536, output_dim=512).to(computation_device)

    final_model = CombinedEmbeddingModel(
        model_main, model_auxiliary, projection_layer, freeze_auxiliary=(arguments.freeze_auxiliary == 'yes')
    ).to(computation_device)

    loss_function = nn.CrossEntropyLoss().to(computation_device)
    optimizer_instance = AdamW(final_model.parameters(), lr=2e-5)

    for epoch_index in range(arguments.epochs):
        train_loss, train_accuracy = train_epoch(final_model, optimizer_instance, loss_function, train_loader, computation_device)
        validation_accuracy = evaluate_model(final_model, validation_loader, computation_device)

        print(f"Epoch {epoch_index + 1}: Training Loss = {train_loss:.4f}, Training Accuracy = {train_accuracy:.4f}, Validation Accuracy = {validation_accuracy:.4f}")

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--epochs', type=int, default=3)
    argument_parser.add_argument('--random_seed', type=int, default=42)
    argument_parser.add_argument('--data_source', type=str, default='gab')
    argument_parser.add_argument('--freeze_auxiliary', type=str, choices=['yes', 'no'], default='yes')
    parsed_arguments = argument_parser.parse_args()

    main(parsed_arguments)

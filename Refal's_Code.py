import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
import csv
import pickle
import sentencepiece
from transformers import DebertaV2Model, DebertaV2Config, DebertaV2Tokenizer

MODEL_NAME = 'microsoft/deberta-v3-base'

# Suppress specific FutureWarnings and DeprecationWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_dataset_with_review_id(file_path):
    texts = []
    review_ids = []
    aspect_labels = []

    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            texts.append(row['Pre_Text'])
            review_ids.append(int(row['Review_ID']))  # Convert to integer
            aspect_labels.append(row['aspectCategory'])

    return texts, review_ids, aspect_labels

def tokenize_text_with_review_id(texts, review_ids, aspect_labels, tokenizer, max_length):
    aspect_label_map = {'food': 0, 'service': 1, 'price': 2, 'ambience': 3, 'views': 4, 'menu': 5, 'staff': 6, 'place': 7, 'drinks': 8, 'location': 9, 'dessert': 10, 'decor': 11, 'clean': 12, 'seating': 13, 'parking': 14}
    aspect_labels = [aspect_label_map[aspect] for aspect in aspect_labels]

    # Ensure texts are converted to strings
    texts = [str(text) for text in texts]

    tokenized_texts = tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = tokenized_texts['input_ids']
    attention_masks = tokenized_texts['attention_mask']
    aspect_labels = torch.tensor(aspect_labels, dtype=torch.long)  # Convert aspect labels to long tensor
    review_ids = torch.tensor(review_ids, dtype=torch.long)  # Convert review IDs to long tensor
    return input_ids, attention_masks, aspect_labels, review_ids

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaModel, DebertaTokenizer

class DeBERTaAspectClassifier(nn.Module):
    def __init__(self, num_aspect_labels=15, hidden_size=768, num_filters=64, filter_sizes=[3, 3], dropout=0.1):
        super(DeBERTaAspectClassifier, self).__init__()
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.aspect_cnn = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, fs) for fs in filter_sizes])  # Conv1d instead of Conv2d
        self.aspect_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128 + hidden_size, num_aspect_labels)  # Adjusted input size
        self.num_aspect_labels = num_aspect_labels
        self.aspect_attn = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        embedded = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        aspect_pooled = []
        for conv in self.aspect_cnn:
            conv_out = F.relu(conv(embedded.permute(0, 2, 1)))  # Adjust input dimensions
            aspect_pooled.append(F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2))
        aspect_concat = torch.cat(aspect_pooled, 1)
        aspect_concat = self.aspect_dropout(aspect_concat)
        # Add aspect attention over DeBERTa embeddings
        attn_weights = F.softmax(self.aspect_attn(embedded), dim=1)
        aspect_context = torch.sum(embedded * attn_weights, dim=1)
        combined = self.fc1(aspect_concat)
        combined = F.relu(combined)
        # Reshape aspect_context to match aspect_concat
        aspect_context = aspect_context.unsqueeze(1).expand(-1, aspect_concat.size(1), -1)
        # Repeat aspect_context along the sequence length
        aspect_context = aspect_context.unsqueeze(2).repeat(1, 1, embedded.size(1), 1)
        combined = torch.cat([combined.unsqueeze(2), aspect_context], dim=3)  # Concatenate along the feature dimension
        # Squeeze the combined tensor to remove extra dimensions
        combined = combined.squeeze(2)

        combined = self.fc2(combined)  # Pass through the final linear layer

        return combined

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, class_weights=None):

    model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, aspect_labels, _ = batch
            input_ids, attention_mask, aspect_labels = input_ids.to(device), attention_mask.to(device), aspect_labels.to(device)
            optimizer.zero_grad()
            aspect_logits = model(input_ids, attention_mask)

            if class_weights is not None:
                aspect_loss = F.cross_entropy(aspect_logits, aspect_labels, weight=class_weights)
            else:
                aspect_loss = F.cross_entropy(aspect_logits, aspect_labels)

            aspect_loss.backward()
            optimizer.step()

        precision_aspect, recall_aspect, f1_aspect = evaluate(model, val_loader, device, model.num_aspect_labels)  # Pass num_aspect_labels to evaluate function
        overall_precision = sum(precision_aspect) / len(precision_aspect)
        overall_recall = sum(recall_aspect) / len(recall_aspect)
        overall_f1 = sum(f1_aspect) / len(f1_aspect)

        # Print the overall precision, recall, and F1-score of the validation data
        print(f'Epoch {epoch+1}/{num_epochs}, Overall Precision: {overall_precision:.3f}, Overall Recall: {overall_recall:.3f}, Overall F1-score: {overall_f1:.3f}')

        # Print the results for each aspect
        for aspect, precision, recall, f1 in zip(range(len(precision_aspect)), precision_aspect, recall_aspect, f1_aspect):
            print(f"Aspect {aspect}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")

def evaluate(model, val_loader, device, num_aspect_labels):
    model.eval()
    aspect_preds = []
    aspect_labels = []
    precision_aspect = []
    recall_aspect = []
    f1_aspect = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, aspect_labels_batch, _ = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            aspect_labels_batch = aspect_labels_batch.to(device)
            aspect_logits = model(input_ids, attention_mask)

            _, predicted_aspect = torch.max(aspect_logits, dim=1)
            aspect_preds.extend(predicted_aspect.tolist())
            aspect_labels.extend(aspect_labels_batch.tolist())

        for aspect_label in range(num_aspect_labels):
            aspect_label_indices = [i for i, label in enumerate(aspect_labels) if label == aspect_label]
            aspect_preds_subset = [aspect_preds[i] for i in aspect_label_indices]
            aspect_labels_subset = [label for i, label in enumerate(aspect_labels) if i in aspect_label_indices]
            precision, recall, f1, _ = precision_recall_fscore_support(aspect_labels_subset, aspect_preds_subset, average='weighted', zero_division=1)
            precision_aspect.append(precision)
            recall_aspect.append(recall)
            f1_aspect.append(f1)

    return precision_aspect, recall_aspect, f1_aspect

import zipfile
def main():
    texts, review_ids, aspect_labels = load_dataset_with_review_id('/content/data.csv')

    train_texts, val_texts, train_review_ids, val_review_ids, train_aspect_labels, val_aspect_labels = train_test_split(texts, review_ids, aspect_labels, test_size=0.2, random_state=42)

    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    max_length = 500
    train_input_ids, train_attention_masks, train_aspect_labels, train_review_ids = tokenize_text_with_review_id(train_texts, train_review_ids, train_aspect_labels, tokenizer, max_length)
    val_input_ids, val_attention_masks, val_aspect_labels, val_review_ids = tokenize_text_with_review_id(val_texts, val_review_ids, val_aspect_labels, tokenizer, max_length)

    print(f"Train Data Size: {len(train_texts)}")
    print(f"Test Data Size: {len(val_texts)}")

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_aspect_labels, train_review_ids)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_aspect_labels, val_review_ids)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = DeBERTaAspectClassifier(num_aspect_labels=15)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 3

    class_counts = np.bincount(train_aspect_labels)
    total_samples = len(train_aspect_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights[class_counts == 0] = 1.0
    class_weights = class_weights / np.sum(class_weights)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    torch.cuda.empty_cache()
    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, class_weights=class_weights_tensor)

    # Save the model
    model_path = '/content/trained_model_aspect_only.pth'
    torch.save(model.state_dict(), model_path)
  
    # Save the model in a zip file
    with zipfile.ZipFile('/content/trained_model_aspect_only.zip', 'w') as zipf:
        zipf.write(model_path, arcname='trained_model_aspect_only.pth')

if __name__ == "__main__":
    main()
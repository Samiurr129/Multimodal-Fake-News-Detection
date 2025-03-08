import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from PIL import Image
from sklearn.model_selection import train_test_split

# Load Dataset (Example: FakeNewsNet or LIAR dataset required)
# Assuming we have a CSV file with columns ['text', 'image_path', 'label']
df = pd.read_csv("fakenews_dataset.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset for Multimodal Learning
class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']
        image_path = row['image_path']
        label = row['label']
        
        # Tokenize text
        text_inputs = self.tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        return text_inputs['input_ids'].squeeze(0), text_inputs['attention_mask'].squeeze(0), image, torch.tensor(label, dtype=torch.long)

# Creating Data Loaders
train_dataset = FakeNewsDataset(train_df, tokenizer, transform)
test_dataset = FakeNewsDataset(test_df, tokenizer, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Multimodal Model (BERT + CNN)
class MultimodalFakeNewsClassifier(nn.Module):
    def __init__(self):
        super(MultimodalFakeNewsClassifier, self).__init__()
        
        # Text Model (BERT)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(768, 256)
        
        # Image Model (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU()
        )
        
        # Fusion & Classification
        self.fc = nn.Linear(512, 2)  # Binary Classification (Real/Fake)
    
    def forward(self, input_ids, attention_mask, images):
        # Text Processing
        text_output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_features = self.text_fc(text_output)
        
        # Image Processing
        image_features = self.cnn(images)
        
        # Fusion
        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.fc(combined_features)
        return output

# Model Training
model = MultimodalFakeNewsClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for input_ids, attention_mask, images, labels in train_loader:
            input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# Start Training
train_model(model, train_loader, criterion, optimizer, epochs=5)

print("Model Training Completed!")


import re
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, Phrases, Phraser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Download required nltk data ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --- Load and preprocess data ---
df = pd.read_csv("labeledTrainData.tsv", sep='\t')
reviews = df['review'].values
labels = df['sentiment'].values

def clean_review(raw_review):
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    review_text = re.sub("[^A-Za-z\s]", " ", review_text)
    review_text = review_text.lower()
    tokens = word_tokenize(review_text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [lemmatizer.lemmatize(w, pos="v") for w in tokens]
    return [w for w in tokens if w not in stop_words and w != ""]

cleaned_reviews = [clean_review(r) for r in reviews]

# --- Create bigrams and trigrams ---
bigram_phraser = Phraser(Phrases(cleaned_reviews, min_count=5, threshold=10))
trigram_phraser = Phraser(Phrases(bigram_phraser[cleaned_reviews], threshold=10))
phrased_reviews = [trigram_phraser[bigram_phraser[r]] for r in cleaned_reviews]

# --- Train Word2Vec ---
embedding_dim = 256
w2v_model = Word2Vec(phrased_reviews, vector_size=embedding_dim, window=5, min_count=3, workers=4)
word_to_idx = {word: idx for idx, word in enumerate(w2v_model.wv.index_to_key)}
embedding_matrix = w2v_model.wv.vectors

# --- Convert text to padded sequences ---
MAX_LEN = 150
PAD_IDX = 0
sequences = []
for review in phrased_reviews:
    seq = [word_to_idx[word] for word in review if word in word_to_idx]
    if len(seq) < MAX_LEN:
        seq += [PAD_IDX] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    sequences.append(seq)

X = np.array(sequences)
y = np.array(labels)

# --- PyTorch Dataset and DataLoader ---
class ReviewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
train_loader = DataLoader(ReviewsDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(ReviewsDataset(X_val, y_val), batch_size=64, shuffle=False)

# --- BiLSTM Model ---
class SentimentBiLSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(SentimentBiLSTM, self).__init__()
        num_words, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(num_words, embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        x = self.dropout1(h_cat)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return torch.sigmoid(self.fc2(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentBiLSTM(embedding_matrix).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(5):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    train_losses.append(total_loss / total)
    train_accs.append(correct / total)
    
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * y_batch.size(0)
            preds = (outputs >= 0.5).float()
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)
    val_losses.append(val_loss / val_total)
    val_accs.append(val_correct / val_total)
    print(f"Epoch {epoch+1}: Train Acc {train_accs[-1]:.4f}, Val Acc {val_accs[-1]:.4f}")

# --- Evaluation ---
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch).squeeze()
        preds = (outputs >= 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
print("Validation Accuracy:", acc)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
plt.title("Confusion Matrix")
plt.show()

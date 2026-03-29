import torch
import sys
import os
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.npy_dataset import NPYDataset
from models.abnormal_activity.cnn_lstm import CNNLSTM

# -------------------------
# Config
# -------------------------
DATA_PATH = "dataset/sequences/data.npy"
LABEL_PATH = "dataset/labels/labels.npy"

BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4

# -------------------------
# Load dataset
# -------------------------
dataset = NPYDataset(DATA_PATH, LABEL_PATH)

# Split (80-20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# -------------------------
# Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNLSTM().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCELoss()

# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(), "models/abnormal_cnn_lstm1.pth")

print("Model saved!")
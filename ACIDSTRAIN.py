# cnn_autoencoder.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # [1, 8, 8] -> [8, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [8, 8, 8] -> [8, 4, 4]
            nn.Conv2d(8, 4, kernel_size=3, padding=1),  # [4, 4, 4]
            nn.ReLU(),
            nn.MaxPool2d(2)  # [4, 4, 4] -> [4, 2, 2]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2),  # [4, 2, 2] -> [8, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),  # [8, 4, 4] -> [1, 8, 8]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_preprocess(path):
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
    ]
    columns += ['label']
    df = pd.read_csv(path, names=columns)
    print(df['label'].unique())
    # 1. Encode categorical columns properly
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure string type
    
    # 2. Handle binary/boolean columns
    bool_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    # 3. Drop unused columns
    df = df.drop(['num_outbound_cmds'], axis=1, errors='ignore')
    
    # 4. Binary label encoding
    df['class'] = df['class'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)
    
    # 5. Scale numerical features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove('class')
    df[numerical_cols] = MinMaxScaler().fit_transform(df[numerical_cols])
    
    # 6. Final conversion check
    X = df.drop(['class'], axis=1).values.astype(np.float32)  # Force float32
    y = df['class'].values
    
    # 7. Reshape to (1,8,8)
    X = np.pad(X, ((0, 0), (0, 64 - X.shape[1])))[:, :64].reshape(-1, 1, 8, 8)
    
    return X, y
X, y = load_preprocess("KDDTrain+.txt")
print("Total samples:", len(X))
print("Normal samples (y=0):", np.sum(y == 0))
print("Attack samples (y=1):", np.sum(y == 1))

if np.sum(y == 0) == 0:
    raise ValueError("No normal samples found! Check label encoding or dataset.")

X_train = torch.tensor(X[y == 0], dtype=torch.float32)  # normal only
print("X_train shape:", X_train.shape)  # Should not be (0, ...)  # normal only
train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)

# Train model
model = CNNAutoencoder()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training CNN Autoencoder...")
for epoch in range(10):
    epoch_loss = 0
    for x in train_loader:
        x = x[0]
        output = model(x)
        loss = criterion(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "cnn_autoencoder_nslkdd.pth")




import yfinance as yf
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

tickers = {
    "stock": "NVDA",       
    "sp500": "^GSPC",      
    "ftse": "^FTSE",      
    "nikkei": "^N225",     
    "vix": "^VIX"          
}

data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start="2025-07-01", end="2026-07-14")
    if "Adj Close" in df.columns:
        data[name]=df["Adj Close"]
    else:
        data[name]=df["Close"]

df_all = pd.concat(data.values(), axis=1, sort=False)
df_all.columns = data.keys()
df_all = df_all.ffill()

for col in df_all.columns:
    for lag in range(1, 6):  # past 5 days
        df_all[f"{col}_lag{lag}"] = df_all[col].shift(lag)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_all[df_all.columns] = scaler.fit_transform(df_all[df_all.columns])

df_all = df_all.dropna()

def create_sequences(data, target_col, seq_length=30):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)].drop(target_col, axis=1).values
        y = data.iloc[i+seq_length][target_col]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(df_all, target_col="stock", seq_length=30)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_p=0.5):
        super(StockPredictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)  
        return out
input_dim = X.shape[2]   
hidden_dim = 24
num_layers = 2
output_dim = 1

model = StockPredictor(input_dim, hidden_dim, num_layers, output_dim, dropout_p=0.5)

model.load_state_dict(torch.load("stock_model_NVDA.pth"))

model.train()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 30
for epoch in range(30):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        if torch.isnan(loss) or torch.isinf(loss):
            print("Invalid loss detected, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "stock_model_NVDA.pth")

model_loaded = StockPredictor(input_dim, hidden_dim, num_layers, output_dim)
model_loaded.load_state_dict(torch.load("stock_model_NVDA.pth"))
model_loaded.eval()

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model_loaded(X_batch)
       
        y_pred_np = y_pred[:5].squeeze().numpy().reshape(-1, 1)
        y_batch_np = y_batch[:5].squeeze().numpy().reshape(-1, 1)
        zeros_pad_pred = np.zeros((y_pred_np.shape[0], df_all.shape[1]-1))
        zeros_pad_actual = np.zeros((y_batch_np.shape[0], df_all.shape[1]-1))

        y_pred_rescaled = scaler.inverse_transform(
            np.hstack([y_pred_np, zeros_pad_pred])
        )[:,0]

        y_actual_rescaled = scaler.inverse_transform(
            np.hstack([y_batch_np, zeros_pad_actual])
        )[:,0]
        print("Predicted (real $):", y_pred_rescaled)
        print("Actual (real $):", y_actual_rescaled)
        break
with torch.no_grad():
    
    last_sequence = df_all.iloc[-30:].drop("stock", axis=1).values
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)  

    next_day_pred = model_loaded(last_sequence)

    next_day_np = next_day_pred.squeeze().numpy().reshape(-1,1)
    zeros_pad = np.zeros((next_day_np.shape[0], df_all.shape[1]-1))

    next_day_rescaled = scaler.inverse_transform(
        np.hstack([next_day_np, zeros_pad])
    )[:,0]

    print("Predicted next day's stock price (real $):", next_day_rescaled[0])










        

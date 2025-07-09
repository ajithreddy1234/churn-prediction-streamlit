import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

data_path = r"C:\Users\venka\PycharmProjects\Churn_analysis\Data\customer_churn_data.csv"
df1 = pd.read_csv(data_path)
df1.drop("CustomerID", axis=1, inplace=True)
df1["Gender"] = LabelEncoder().fit_transform(df1["Gender"])
df1["InternetService"] = df1["InternetService"].fillna("Unknown")
df1= pd.get_dummies(df1, columns=["InternetService"], drop_first=True)
df1= pd.get_dummies(df1, columns=["ContractType"], drop_first=True)
df1["TechSupport"] = LabelEncoder().fit_transform(df1["TechSupport"])
df1["Churn"] = df1["Churn"].map({"Yes": 1, "No": 0})
bool_cols = df1.select_dtypes(include='bool').columns
df1[bool_cols] = df1[bool_cols].astype(int)
scaler = StandardScaler()
scale_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
df1[scale_cols] = scaler.fit_transform(df1[scale_cols])
print(df1.columns)
x=df1[['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TotalCharges',
       'TechSupport', 'InternetService_Fiber Optic',
       'InternetService_Unknown', 'ContractType_One-Year',
       'ContractType_Two-Year']]
y=df1[["Churn"]]

X = x.values.astype(np.float32)
Y = y.values.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=42)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
X_train = torch.tensor(X_train_balanced, dtype=torch.float32)
y_train = torch.tensor(y_train_balanced, dtype=torch.float32)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
model = ANN()
batch_size = 64
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch.squeeze())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch).squeeze()
        preds = (outputs >= 0.5).float().cpu().numpy()
        labels = y_batch.squeeze().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
print(classification_report(all_labels, all_preds, target_names=["No Churn", "Churn"]))
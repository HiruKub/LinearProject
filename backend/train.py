import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

csv_file = pd.read_csv("data/diabetes.csv")

feature = ["Glucose", "BloodPressure", "BMI", "Age","DiabetesPedigreeFunction"]
x = csv_file[feature].to_numpy()
y = csv_file["Outcome"].to_numpy()

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(1)

torch.manual_seed(42)

num_samples = x_tensor.shape[0]

indices = torch.randperm(num_samples)

# (Train 80%, Test 20%)
train_size = int(0.8 * num_samples)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

x_train = x_tensor[train_indices]
y_train = y_tensor[train_indices]

x_test = x_tensor[test_indices]
y_test = y_tensor[test_indices]

# mean and std for normalization
mean = x_train.mean(dim=0)  # dim=0 calculate with column
std = x_train.std(dim=0)

# data scaling(standardization)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

nodes = 5
class DiabetesNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, nodes),  # W1 5xnodes and bias1
            nn.ReLU(),  # Activation function
            nn.Linear(nodes, 1),  # W2 nodesx1 and bias2
            nn.Sigmoid(),  # result 0 - 1
        )

    def forward(self, x):
        return self.model(x)


model = DiabetesNN()

criterion = nn.BCELoss() #create criterion
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

epochs = 1000
for epoch in range(epochs):
    # forward pass
    y_predict = model(x_train)

    loss_fn = criterion(y_predict, y_train)

    # clear old gradient
    optimizer.zero_grad()

    # Backpropagation
    loss_fn.backward()

    # scaling weight and bias
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_fn.item():.4f}")

with torch.no_grad():
    y_predict_test = model(x_test)
    predicted_classes = (y_predict_test >= 0.5).float()
    accuracy = (predicted_classes == y_test).float().mean()
    print(f"\nTest Accuracy: {accuracy.item() * 100:.2f}%")

torch.save({
    "model_state": model.state_dict(),
    "mean": mean,
    "std": std,
    "features": feature
}, f"backend/diabetes_model(5x{nodes}).pth")

print("model saved")
print("ค่า Weights")
print(f"Weight Matrix W1 (5x{nodes}):\n", model.model[0].weight.data)
print(f"Bias Vector b1 ({nodes}):\n", model.model[0].bias.data)
print(f"Weight Matrix W2 (1x{nodes}):\n", model.model[2].weight.data)
print("Bias b2:\n", model.model[2].bias.data)
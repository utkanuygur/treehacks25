from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from pyngrok import ngrok  # Import ngrok

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the neural network
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Single logit
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set input dimension based on your dataset
input_dim = 30

# Load the trained model and move it to GPU if available
model = Net(input_dim).to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Define input structure
class PredictionInput(BaseModel):
    features: list[float]  # List of feature values

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Convert input data to tensor and move to GPU
    features_tensor = torch.tensor(input_data.features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(features_tensor).squeeze(1)
        probability = torch.sigmoid(output).item()
        predicted_class = 1 if probability > 0.5 else 0
    
    return {"probability": probability, "predicted_class": predicted_class}

if __name__ == "__main__":
    import uvicorn

    # Start Uvicorn server in the background
    port = 8000
    public_url = ngrok.connect(port).public_url
    print(f"Public URL: {public_url}")

    uvicorn.run(app, host="0.0.0.0", port=port)

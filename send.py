import requests

url = "https://ce7c-68-65-164-53.ngrok-free.app/predict"  # Change this to your ngrok URL if needed
data = {
    "features": [21,0,0,203,0,3,0,0,0,3,0,3,0,0,0,0,0,0,0,0,1,3,1,3,0,0,3,0,0,0]  # Replace with actual feature values
}

response = requests.post(url, json=data)
print(response.content) 


# 
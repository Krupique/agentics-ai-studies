import requests
import json

print("The Agent team is processing your request. Please wait...\n")

# API URL
API_URL = "http://localhost:8000/execute"

# Topic
data = {"topic": "Pretrial detention and post-conviction incarceration pending appeal in U.S. law"}

# Request via POST to API
response = requests.post(API_URL, json=data)

# Checking if the requistion was successfuly
if response.status_code == 200:
    result = response.json()
    
    # Get the generated report
    if "result" in result and "raw" in result["result"]:
        report = result["result"]["raw"]
        
        print("\n*****Report:*****\n")
        print(report)
    
    else:
        print("Error: Unespected answer from API.")
else:
    print(f"Request error: {response.status_code} - {response.text}")

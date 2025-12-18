import requests
import json

# Test data for water quality prediction
# NOTE: The HDBSCAN model classifies data as:
# - Cluster 0 or 1 = POTABLE (data within training distribution)
# - Cluster -1 (noise/outlier) = NON POTABLE (extreme outliers)
test_cases = [
    {
        "name": "Test 1: EXTREME OUTLIER - All values very negative (should be NON POTABLE)",
        "data": {
            "ph": -50,              # Impossible negative pH
            "hardness": -500,       # Impossible negative
            "solids": -50000,       # Impossible negative
            "chloramines": -50,     # Impossible negative
            "sulfate": -500,        # Impossible negative
            "conductivity": -800,   # Impossible negative
            "organic_carbon": -100, # Impossible negative
            "trihalomethanes": -200,# Impossible negative
            "turbidity": -50        # Impossible negative
        }
    },
    {
        "name": "Test 2: EXTREME OUTLIER - All values extremely high (should be NON POTABLE)",
        "data": {
            "ph": 50,               # Impossible high pH
            "hardness": 5000,       # Extremely high
            "solids": 500000,       # Extremely high
            "chloramines": 100,     # Extremely high
            "sulfate": 5000,        # Extremely high
            "conductivity": 10000,  # Extremely high
            "organic_carbon": 500,  # Extremely high
            "trihalomethanes": 1000,# Extremely high
            "turbidity": 100        # Extremely high
        }
    },
    {
        "name": "Test 3: EXTREME OUTLIER - Mixed extreme values (should be NON POTABLE)",
        "data": {
            "ph": 0,                # Zero pH (impossible)
            "hardness": 0,
            "solids": 0,
            "chloramines": 0,
            "sulfate": 0,
            "conductivity": 0,
            "organic_carbon": 0,
            "trihalomethanes": 0,
            "turbidity": 0
        }
    },
    {
        "name": "Test 4: Normal water within training range (should be POTABLE)",
        "data": {
            "ph": 7.0,              # Neutral pH - good
            "hardness": 200,        # Normal hardness
            "solids": 20000,        # Normal solids
            "chloramines": 7,       # Normal chloramines
            "sulfate": 330,         # Normal sulfate
            "conductivity": 420,    # Normal conductivity
            "organic_carbon": 14,   # Normal organic carbon
            "trihalomethanes": 66,  # Normal THMs
            "turbidity": 4          # Clear water
        }
    },
    {
        "name": "Test 5: Slightly bad but in range (likely POTABLE - model limitation)",
        "data": {
            "ph": 4.0,              # Acidic but in range
            "hardness": 350,        # High
            "solids": 45000,        # High
            "chloramines": 12,      # High
            "sulfate": 450,         # High
            "conductivity": 700,    # High
            "organic_carbon": 25,   # High
            "trihalomethanes": 120, # High
            "turbidity": 8          # Turbid
        }
    }
]

print("=" * 60)
print("WATER QUALITY PREDICTION TEST")
print("=" * 60)

for test in test_cases:
    print(f"\n{test['name']}")
    print("-" * 50)
    print(f"Input data: {json.dumps(test['data'], indent=2)}")
    
    try:
        response = requests.post(
            'http://localhost:5000/predict_water', 
            json=test['data'], 
            timeout=10
        )
        result = response.json()
        
        if result.get('success'):
            print(f"\n>>> RESULT: {result.get('label')}")
            print(f">>> Is Potable: {result.get('is_potable')}")
            print(f">>> Cluster: {result.get('cluster')}")
            print(f">>> Confidence: {result.get('confidence')}")
            print(f">>> Explanation: {result.get('explanation')}")
        else:
            print(f"ERROR: {result.get('error')}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Server not running. Please start the Flask app first with: python app.py")
        break
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

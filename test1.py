import json
import os

DATA_DIR = r"C:\Users\bapun\Downloads\Order_approval\data"
TABLES_JSON_PATH = os.path.join(DATA_DIR, "financial_tables.json")

# Check if the file exists
if not os.path.exists(TABLES_JSON_PATH):
    print("❌ Financial tables JSON file not found!")
else:
    # Load JSON
    with open(TABLES_JSON_PATH, "r") as f:
        data = json.load(f)
    
    if len(data) == 0:
        print("❌ No structured financial data found!")
    else:
        print(f"✅ Found {len(data)} structured financial table entries.")

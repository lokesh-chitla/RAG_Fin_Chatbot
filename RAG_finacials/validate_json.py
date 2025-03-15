import json
import os

DATA_DIR = r"C:\Users\bapun\Downloads\Order_approval\data"
TABLES_JSON_PATH = os.path.join(DATA_DIR, "financial_tables.json")

# Load the JSON file
with open(TABLES_JSON_PATH, "r") as f:
    try:
        data = json.load(f)
        print(f"✅ Loaded JSON with {len(data)} entries.")
        
        # Check for malformed entries
                # Check for valid dictionary structure
        valid_entries = []
        for entry in data:
            if isinstance(entry, dict) and "description" in entry and "value" in entry:
                valid_entries.append(entry)
            else:
                print(f"⚠️ Skipping entry (may still contain useful data): {entry}")


        # Overwrite the file with cleaned data
        with open(TABLES_JSON_PATH, "w") as out_f:
            json.dump(valid_entries, out_f, indent=4)
        
        print(f"✅ Cleaned JSON saved with {len(valid_entries)} valid entries.")
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")

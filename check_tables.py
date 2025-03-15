import json
import os

# **🔹 Path to JSON file containing extracted financial tables**
TABLES_JSON_PATH = r"C:\Users\bapun\Downloads\ORDER_APPROVAL\data\financial_tables.json"

# **🔹 Check if JSON file exists**
if os.path.exists(TABLES_JSON_PATH):
    with open(TABLES_JSON_PATH, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    # **🔹 Print the extracted table data in readable format**
    print("✅ Loaded Financial Tables Data:")
    print(json.dumps(tables_data, indent=4))  # Pretty-print JSON

    # **🔹 Check if tables are actually stored**
    if not tables_data:
        print("⚠️ No tables found in the JSON file! `embedder.py` may not be extracting tables correctly.")
    else:
        print(f"✅ Found {len(tables_data)} table entries.")
else:
    print(f"❌ File not found: {TABLES_JSON_PATH}\nRun `embedder.py` first!")

import sqlite3
import xml.etree.ElementTree as ET
import json

def create_database_from_json(json_file, db_name="data.db"):
    with open(json_file, "r") as file:
        data = json.load(file)
    
    # Determine the data structure
    if isinstance(data, dict):
        # If it's a dictionary, assume it contains a key pointing to the list
        for key in data:
            if isinstance(data[key], list) and all(isinstance(item, dict) for item in data[key]):
                data = data[key]
                break
        else:
            print("⚠ No valid list of records found in JSON.")
            return
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        # If it's already a list of dictionaries, use it directly
        pass
    else:
        print("⚠ Unsupported JSON structure.")
        return
    
    # Extract column names dynamically from the first item
    first_record = data[0]
    columns = [f"{key} TEXT" for key in first_record.keys()]
    columns_sql = ", ".join(columns)
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns_sql}
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database and table dynamically created!")

def insert_data_dynamic(db_name, data):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    for record in data:
        keys = ", ".join(record.keys())
        placeholders = ", ".join(["?" for _ in record.keys()])
        values = tuple(record.values())
        
        cursor.execute(f"INSERT INTO records ({keys}) VALUES ({placeholders})", values)
    
    conn.commit()
    conn.close()
    print("✅ Data inserted dynamically!")

def parse_json_and_store(json_file, db_name="data.db"):
    with open(json_file, "r") as file:
        data = json.load(file)
    
    # Determine the correct structure of the data
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], list) and all(isinstance(item, dict) for item in data[key]):
                data = data[key]
                break
    elif not (isinstance(data, list) and all(isinstance(item, dict) for item in data)):
        print("⚠ Unsupported JSON structure.")
        return
    
    insert_data_dynamic(db_name, data)


if __name__ == "__main__":
    json_file = "data/rso_data/rso_data.json" # Change this to your JSON file path
    db_name = "data/rso_data/rso_data.db"     # Change database name if needed
    
    create_database_from_json(json_file, db_name)
    parse_json_and_store(json_file, db_name)
    print("✅ JSON Data dynamically stored in the database!")

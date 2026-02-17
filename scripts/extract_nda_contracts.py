# extract_nda_contracts.py

"""
This script extracts Non-Disclosure Agreements (NDA) from the CUAD dataset with annotations.
"""

import json
import os

# Function to load CUAD dataset

def load_cuad_dataset(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to extract NDA contracts

def extract_nda_contracts(data):
    nda_contracts = []
    for item in data:
        if 'NDA' in item['contract_type']:
            nda_contracts.append(item)
    return nda_contracts

# Main function

def main():
    dataset_path = os.path.join(os.getcwd(), 'path_to_cuad_dataset.json')  # Update this path
    cuad_data = load_cuad_dataset(dataset_path)
    nda_contracts = extract_nda_contracts(cuad_data)
    print(f'Extracted {len(nda_contracts)} NDA contracts.')

if __name__ == '__main__':
    main()
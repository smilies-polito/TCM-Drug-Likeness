import requests
from bs4 import BeautifulSoup
import certifi
import csv
import concurrent.futures
from tqdm import tqdm
from threading import Lock
import time
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("terminal_output.txt"),
    logging.StreamHandler()
])

# Base URL to access substance pages
base_url = "https://zinc15.docking.org/substances/"

# List to store the results
results = []
lock = Lock()

def get_smiles(zinc_id, label):
    """
    Fetches the SMILES string for a given ZINC ID from the ZINC database.

    Args:
        zinc_id (str): The ZINC ID of the compound.
        label (str): A label associated with the compound.

    Returns:
        tuple: A tuple containing the ZINC ID, the SMILES string (or None if not found), and the label.

    Raises:
        requests.RequestException: If there is an issue with the HTTP request.

    This function attempts to retrieve the SMILES string for a given ZINC ID by making an HTTP request to the ZINC database.
    It retries the request up to 5 times in case of failure, with a 1-second delay between attempts.
    If the SMILES string is successfully retrieved, it is returned along with the ZINC ID and label.
    If the SMILES string is not found or the request fails after 5 attempts, None is returned for the SMILES string.
    """
    url = base_url + zinc_id + "/"
    retries = 5
    for attempt in range(retries):
        try:
            response = requests.get(url, verify=certifi.where(), timeout=10)  # Set a timeout of 10 seconds
            response.raise_for_status()  # Verify that the request was executed successfully
            soup = BeautifulSoup(response.content, 'html.parser')
            smiles_field = soup.find('input', {'id': 'substance-smiles-field'})
            if smiles_field and 'value' in smiles_field.attrs:
                if attempt > 0:
                    logging.info(f"Successfully retrieved SMILES for {zinc_id} on attempt {attempt + 1}")
                return (zinc_id, smiles_field['value'], label)
            else:
                logging.warning(f"SMILES not found for {zinc_id}")
                return (zinc_id, None, label)
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed for {zinc_id}. Error: {e}")
            time.sleep(1)  # Wait 1 second before trying again
    logging.error(f"Failed to retrieve page for {zinc_id} after {retries} attempts.")
    return (zinc_id, None, label)

# Function to handle results in a thread-safe manner
def handle_result(future):
    result = future.result()
    with lock:
        results.append(result)

def main(dataset_type):
    # Input and output files
    input_file = f'ZINC\\{dataset_type}_extracted_zinc_ids_new.csv'
    output_file = f'ZINC\\ZINC_{dataset_type}_nuovi_samples_smiles_output.csv'

    # Read the input CSV file and create a list of ZINC IDs and labels
    zinc_data = []
    with open(input_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            zinc_data.append((row[0], row[1]))  # Add tuples (ZINCID, label)

    # Use multithreading to perform scraping
    num_threads = 50
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(get_smiles, zinc_id, label) for zinc_id, label in zinc_data]
        # Use tqdm to show the progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Progress"):
            handle_result(future)

    # Write the results into a new CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['ZINCID', 'Smiles', 'Drug'])  # Header
        csvwriter.writerows([result for result in results if result[1] is not None])

    logging.info("Scraping complete. Results saved to %s", output_file)

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["in_vivo", "in_vitro"]:
        print("Usage: python ZINC_take_smiles.py [in_vivo|in_vitro]")
        sys.exit(1)
    dataset_type = sys.argv[1]
    main(dataset_type)

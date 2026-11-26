import requests
from bs4 import BeautifulSoup
import certifi
import re
import concurrent.futures

# Base URLs for in-vivo and in-vitro datasets
base_urls = {
    "in-vivo": "https://zinc20.docking.org/substances/subsets/in-vivo/?page=",
    "in-vitro": "https://zinc20.docking.org/substances/subsets/in-vitro/?page="
}

# Send GET requests to all the pages with SSL verification using certifi
zinc_ids = []

def scrape_page(page, base_url):
    """
    Scrapes a specific page for ZINC IDs.
    Args:
        page (int): The page number to scrape.
        base_url (str): The base URL to use for scraping.
    Returns:
        None
    This function constructs a URL using the base URL and the given page number,
    sends a GET request to the URL, and parses the HTML content using BeautifulSoup.
    It searches for all anchor tags with href and title attributes, and then
    extracts ZINC IDs from the text of these links using a regular expression.
    If the request fails or no ZINC IDs are found, it prints an error message.
    """
    url = base_url + str(page)
    response = requests.get(url, verify=certifi.where())
    print(f"Scraping page {page}...")
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        all = soup.find_all('a', href=True, title=True)
        # Find all the elements that match the pattern of the ZINC ID links
        matches = []
        for link in all:
            match = re.search(r'ZINC\d+', link.text)
            matches.append(match)
            if match:
                zinc_ids.append(match.group())

        if len(matches) == 0:
            print(f"Failed to retrieve ZINC IDs from page {page}.")
        
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

def main(dataset_type):
    """
    Main function to scrape ZINC IDs based on the dataset type.
    Args:
        dataset_type (str): The type of dataset to scrape ("in-vivo" or "in-vitro").
    Returns:
        None
    """
    if dataset_type not in base_urls:
        print(f"Invalid dataset type: {dataset_type}. Please choose 'in-vivo' or 'in-vitro'.")
        return

    base_url = base_urls[dataset_type]

    # Use multithreading to scrape multiple pages concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Scrape pages 1 to 1296
        executor.map(lambda page: scrape_page(page, base_url), range(1, 1297))

    # Print the extracted ZINC IDs
    print(f"Extracted {len(zinc_ids)} ZINC IDs.")

    # Save the ZINC IDs to a CSV file
    output_filename = f'{dataset_type}_extracted_zinc_ids.csv'
    with open(output_filename, 'w') as output_file:
        for zinc_id in zinc_ids:
            output_file.write(zinc_id + '\n')

if __name__ == "__main__":
    dataset_type = input("Enter the dataset type (in-vivo or in-vitro): ").strip()
    main(dataset_type)

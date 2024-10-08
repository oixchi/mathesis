import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import re
from nltk.stem import PorterStemmer

def get_ebrains_links(base_url):
    # Send an HTTP request to the webpage
    response = requests.get(base_url)
    # Create a BeautifulSoup object to parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all anchor tags
    all_links = soup.find_all('a', href=True)
    # Extract links that start with 'ebrains.eu' or are relative links
    filtered_links = []
    
    for link in all_links:
        href = link['href']
        # Join relative links with the base URL
        full_url = urljoin(base_url, href)
        
        # Check if the link starts with 'https://www.ebrains.eu/' or contains 'ebrains.eu'
        if 'ebrains.eu' in full_url:
            filtered_links.append(full_url)
    
    # Remove duplicates and return the filtered links
    unique_links = list(set(filtered_links))
    return unique_links

def scrape_ebrains_links(ebrains_links, output_file="ebrains_dataset.txt"):
    # Start time
    start_time = time.time()
    print(f"Function call started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    for url in ebrains_links:
        try:
            print(f"Scraping URL: {url}")
            # Send a GET request to the URL
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Failed to retrieve {url}. Status code: {response.status_code}")
                continue  # Skip to the next URL

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Append the URL as a header in the output file
            with open(output_file, "a", encoding="utf-8") as file:        
                for tag in soup.find_all(['p', 'div', 'h1', 'h2', 'h3']):
                    if tag.name == 'div':
                        for p_tag in tag.find_all('p'):
                            file.write(p_tag.get_text().strip() + "\n")
                    else:
                        file.write(tag.get_text().strip() + "\n")
                    
            # Optional: Add a delay between requests to avoid overwhelming the server
            time.sleep(2)

        except Exception as e:
            print(f"An error occurred while scraping {url}: {e}")

    # End time
    end_time = time.time()
    print(f"Function call scrape links ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60

    # Print elapsed time
    print(f"Total time taken: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)")

def remove_duplicate_lines(input_file):
    # Use a set to store unique lines
    seen_lines = set()
    unique_lines = []

    # Read the input file and add lines to the set
    with open(input_file, 'r', encoding='utf-8') as file:
       for line in file:
            stripped_line = line.strip()  # Remove leading and trailing whitespace
            if stripped_line not in seen_lines:
                seen_lines.add(stripped_line)  # Mark the line as seen
                unique_lines.append(stripped_line)  # Keep the first occurrence

    # Write the unique lines back to the output file
    with open(input_file, 'w', encoding='utf-8') as file:
        for line in unique_lines:
            file.write(line + '\n')  # Write each unique line

def clean_text(data):
    # Initialize stemmer
    print("Function called")
    stemmer = PorterStemmer()
    cleaned_data = []

    for line in data:
        # Remove leading and trailing whitespace
        line = line.strip()
        # Convert to lowercase
        line = line.lower()        
        # Remove punctuation
        line = re.sub(r'[^\w\s]', '', line)        
        # Remove extra spaces
        line = re.sub(r'\s+', ' ', line)
        # Tokenization (split into words)
        tokens = line.split()

        # Remove stop words and apply stemming
        tokens = [stemmer.stem(word) for word in tokens]

        # Join tokens back to a single string
        cleaned_line = ' '.join(tokens)

        if cleaned_line:  # Ensure we don't add empty lines
            cleaned_data.append(cleaned_line)

    return cleaned_data


def write_clean_dataset():
    # Read dataset
    input_file = 'ebrains_dataset.txt'  # Replace with your input file path
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.readlines()

    # Clean the dataset
    cleaned_data = clean_text(data)

    # Optionally, write cleaned data to a new file
    output_file = 'cleaned_dataset.txt'  # Replace with desired output file path
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in cleaned_data:
            file.write(line + '\n')


def main():
    base_url = 'https://www.ebrains.eu/'
    ebrains_links = sorted(get_ebrains_links(base_url))
    scrape_ebrains_links(ebrains_links, output_file="ebrains_dataset_new.txt")


if __name__ == "__main__":
    # This condition ensures that the main() function
    # only runs if this script is executed directly,
    # and not when it's imported as a module in another script.
    main()
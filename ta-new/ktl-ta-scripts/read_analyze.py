import requests

# Path to the text file you want to analyze
text_file_path = "sample.txt"

# FastAPI endpoint URL
url = "http://localhost:8000/analyze"

def read_file(file_path):
    """
    Read the contents of a text file.

    Returns:
        lines (list): A list of strings from the file, split by line.
    """
    # Open file and read lines
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Clean up lines (strip empty spaces and remove newlines)
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def analyze_texts(texts):
    """
    Send the list of texts to the FastAPI /analyze endpoint.

    Args:
        texts (list): List of strings to analyze.

    Returns:
        dict: The API response (predictions for each text).
    """
    # Prepare payload for POST request
    payload = {"texts": texts}

    # Send POST request to the FastAPI server
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        # Success response: return the result
        return response.json()
    else:
        # Error: Print the problem and return None
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    # Read texts from the file
    print("Reading file...")
    texts = read_file(text_file_path)
    print(f"File content: {texts}")

    # Send the texts to FastAPI for analysis
    print("Analyzing texts...")
    results = analyze_texts(texts)
    print("Results:")
    if results:
        print(results)

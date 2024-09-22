import requests
import json

def coreference_resolution(text):
    # Implement coreference resolution algorithm here
    url = "http://localhost:6006/coreference"
    data = {"text_block": text}
    
    try:
        # Make HTTP POST request with JSON data
        response = requests.post(url, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            
            # Assuming the API returns a 'resolved_text' field
            resolved_text = result.get('text_block', text)
            return resolved_text
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            return text
    
    except requests.RequestException as e:
        print(f"Error: {e}")
        return text

if __name__ == "__main__":
    # Example usage
    text = "John went to the store. He bought some milk. The store was closed when he arrived."
    resolved_text = coreference_resolution(text)
    print("Original text:", text)
    print("Resolved text:", resolved_text)

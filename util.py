import requests
import json
import os
from tenacity import retry, stop_after_delay, wait_exponential
import sentry_sdk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Sentry
sentry_sdk.init(os.getenv("SENTRY_DSN"))

def handle_response(response, text):
    if response.status_code == 200:
        result_data = response.json()
        print("Response from server:", result_data)
        return result_data.get("text", text)
    else:
        print(f"Failed to get valid response, status code: {response.status_code}")
        return text  # Return original chunks if there's an error

# Log retries to Sentry before sleeping
def log_retry_to_sentry(retry_state):
    error_message = (
        f"Retrying {retry_state.fn.__name__} after {retry_state.outcome.exception()} "
        f"for the {retry_state.attempt_number} time. Sleeping for {retry_state.next_action.sleep} seconds."
    )
    sentry_sdk.capture_message(error_message)

# Retry for up to 3 days with exponential backoff
@retry(
    stop=stop_after_delay(60 * 60 * 24 * 3),  # Retry for up to 3 days
    wait=wait_exponential(multiplier=1, min=4, max=3600),  # Exponential backoff between 4s to 1 hour
    before_sleep=log_retry_to_sentry  # Log retries to Sentry before sleep
)
def http_put(url, chunks, handle_response_func=None):
    # Data to send in the PUT request
    data = {
        "code": os.getenv("API_ENCRYPTION_CODE"),
        "chunks": chunks
    }
    
    # Headers for the request
    headers = {'Content-Type': 'application/json'}
    
    # Make the PUT request
    response = requests.put(url, headers=headers, data=json.dumps(data))

    # Use the custom response handler if provided, else default
    if handle_response_func:
        return handle_response_func(response, chunks)
    else:
        return handle_response(response, chunks)

# Example usage
if __name__ == "__main__":
    url = 'http://example.com/api/resource/1'  # Ensure the ID is included in the URL
    chunks = ["chunk1", "chunk2"]  # Replace with actual chunked data

    try:
        result = http_put(url, chunks)
        print("Final result:", result)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Error after retries: {e}")

import os
import tiktoken
from typing import List, Dict

ai_model = os.getenv("OPENAI_MODEL_70B")

def count_message_tokens(messages: List[Dict[str, str]], model: str = ai_model) -> int:
    """
    Count the number of tokens in a list of messages for the specified OpenAI model.

    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries.
        model (str): The name of the OpenAI model to use for token counting.
                     Defaults to "gpt-3.5-turbo-0613".

    Returns:
        int: The total number of tokens in the messages.

    Raises:
        ValueError: If an unsupported model is specified.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model '{model}' not found. Using default 'gpt-3.5-turbo-0613' encoding.")
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")

    if model.startswith("gpt-3.5-turbo"):
        return count_messages_tokens(messages, encoding)
    elif model.startswith("gpt-4"):
        return count_messages_tokens(messages, encoding)
    else:
        raise ValueError(f"Unsupported model: {model}")

def count_messages_tokens(messages: List[Dict[str, str]], encoding: tiktoken.Encoding) -> int:
    """
    Count tokens for a list of messages with a specific encoding.
    """
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # If there's a name, the role is omitted
                num_tokens -= 1  # Role is always required and always 1 token
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens

def count_tokens(text: str, model: str = ai_model) -> int:
    """
    Count the number of tokens in the given text for the specified OpenAI model.

    Args:
        text (str): The input text to count tokens for.
        model (str): The name of the OpenAI model to use for token counting. 
                     Defaults to "gpt-3.5-turbo".

    Returns:
        int: The number of tokens in the input text.

    Raises:
        ValueError: If an unsupported model is specified.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model '{model}' not found. Using default 'gpt-3.5-turbo' encoding.")
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    return len(encoding.encode(text))

# Example usage
if __name__ == "__main__":
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
    
    token_count = count_message_tokens(sample_messages)
    print(f"The messages contain {token_count} tokens.")

    # Example with a different model
    gpt4_token_count = count_message_tokens(sample_messages, "gpt-4-0613")
    print(f"For GPT-4, the messages contain {gpt4_token_count} tokens.")

    sample_text = "This is a sample text to count tokens for OpenAI API calls."
    token_count = count_tokens(sample_text)
    print(f"The text contains {token_count} tokens.")

    # Example with a different model
    gpt4_token_count = count_tokens(sample_text, "gpt-4")
    print(f"For GPT-4, the text contains {gpt4_token_count} tokens.")

"""
single_call_comment_search.py

Two main parts:

1) The function get_top_10_relevant_comments(file_path, search_string):
   - Reads the .txt file of comments
   - Calls Perplexity API with a single chat-completion request
   - Returns a list of up to 10 most relevant comments (strings only)

2) A main() function for local testing:
   - Parses command-line args
   - Calls get_top_10_relevant_comments()
   - Prints the resulting top comments
"""

import sys
import argparse
import json
import requests

# -----------------------------------------------------------------
# Hard-code your Perplexity API key (replace with your actual token)
# -----------------------------------------------------------------

"""
@param: file_path, the name of the .txt file to be inputted
@param: search_string, a string to be searched
@output: a list of top 10 relevant comments
"""
def get_top_10_relevant_comments(file_path: str, search_string: str) -> list:
    """
    Reads a .txt file of comments, sends a single request to Perplexity,
    and returns the top 10 most relevant comments as a list of strings.
    """
    API_KEY = "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"

    # Perplexity Chat Completions endpoint
    PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

    # Read comments from file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            comments = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    if not comments:
        # If no comments, return an empty list
        return []

    # ---------------------------------------------------------------
    # Build a single prompt listing all comments with an index
    # ---------------------------------------------------------------
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant that ONLY returns JSON in the exact format specified. "
            "No additional commentary. No markdown. No code blocks. "
            "We want the top 10 relevant comments in descending order of relevance."
        )
    }

    enumerated_comments = "\n".join(f"{idx+1}) {comment}" for idx, comment in enumerate(comments))

    user_prompt = (
        f"The user query is: '{search_string}'.\n\n"
        f"Here is the list of comments:\n{enumerated_comments}\n\n"
        "Return exactly this JSON format:\n\n"
        "{\"ranked_comments\": [\n"
        "  {\"index\": <integer>, \"comment\": \"<text>\"},\n"
        "  ... up to 10 items in total ...\n"
        "]}\n\n"
        "The 'index' is the original line number in the list above (1-based), and 'comment' is the comment text.\n"
        "List only the top 10 relevant comments, in descending order (most relevant first). If fewer than 10 comments, list them all.\n"
        "DO NOT RETURN ANYTHING ELSE."
    )

    user_message = {
        "role": "user",
        "content": user_prompt
    }

    payload = {
        "model": "sonar",  # or other Perplexity models
        "messages": [system_message, user_message],
        "max_tokens": 500,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 0,
        "frequency_penalty": 1,
        "presence_penalty": 0,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Make the API request
    try:
        response = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request to Perplexity failed: {e}")

    # Check HTTP status
    if response.status_code != 200:
        raise RuntimeError(
            f"Perplexity API returned status code {response.status_code}.\n"
            f"Response Body: {response.text}"
        )

    # Parse JSON response
    try:
        resp_json = response.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"Could not parse JSON from Perplexity response.\nRaw: {response.text}")

    choices = resp_json.get("choices", [])
    if not choices:
        raise RuntimeError("No 'choices' returned by the Perplexity model.")

    model_content = choices[0].get("message", {}).get("content", "").strip()
    if not model_content:
        raise RuntimeError("The model returned empty content.")

    # Attempt to parse the content as JSON
    try:
        result_json = json.loads(model_content)
    except json.JSONDecodeError:
        raise RuntimeError(f"Model content is not valid JSON:\n{model_content}")

    # We expect {"ranked_comments": [ {...}, {...}, ... ]}
    ranked_comments = result_json.get("ranked_comments", [])
    if not isinstance(ranked_comments, list):
        raise RuntimeError("'ranked_comments' is missing or not a list.")

    # Convert the returned items to just the comment text
    top_comments = []
    for item in ranked_comments:
        comment_text = item.get("comment", "")
        if comment_text:
            top_comments.append(comment_text)

    return top_comments

def main():
    """
    Command-line test harness:
    1) Parses command-line arguments.
    2) Calls get_top_10_relevant_comments().
    3) Prints the list of up to 10 relevant comments.
    """
    parser = argparse.ArgumentParser(description='Batch comment relevance check (single Perplexity call).')
    parser.add_argument('file_path', type=str, help='Path to the .txt file containing comments.')
    parser.add_argument('search_string', type=str, help='A query string for relevance.')
    args = parser.parse_args()

    try:
        relevant_comments = get_top_10_relevant_comments(args.file_path, args.search_string)
    except Exception as e:
        print("[Error]", e)
        sys.exit(1)

    if not relevant_comments:
        print("No relevant comments found (or no comments in file).")
        return

    print("\nTop Relevant Comments:\n")
    for idx, comment in enumerate(relevant_comments, start=1):
        print(f"{idx}. {comment}")

if __name__ == "__main__":
    main()
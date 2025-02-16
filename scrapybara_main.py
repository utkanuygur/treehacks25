from scrapybara import Scrapybara
from scrapybara.anthropic import Anthropic
from scrapybara.tools import ComputerTool, BashTool, EditTool
from scrapybara.prompts import UBUNTU_SYSTEM_PROMPT, BROWSER_SYSTEM_PROMPT
from dotenv import load_dotenv
import os
import time
from playwright.sync_api import sync_playwright


def send_email_via_outlook():
    """
    Uses Playwright to log in to Microsoft Outlook via its web interface,
    then leverages Scrapybara's agent to compose and send an email.
    
    Steps:
      1. Start an Ubuntu instance and a browser session.
      2. Use Playwright to navigate to the Outlook login page and log in
         with the account DUTYtreehacks@outlook.com (using the password from the environment).
      3. Once logged in, use Scrapybara's agent (via available tools) to:
         - Click the "New message" button.
         - Fill in the recipient, subject, and email body.
         - Click "Send" to dispatch the email.
    """
    # Load environment variables (ensure SCRAPY_KEY and OUTLOOK_PASSWORD are defined in your .env file)
    load_dotenv()
    client = Scrapybara(api_key=os.getenv("SCRAPY_KEY"))
    outlook_password = os.getenv("OUTLOOK_PASSWORD")
    
    # Start an Ubuntu instance.
    instance = client.start_ubuntu()
    print("Ubuntu instance started.")
    
    # Start a browser session via Scrapybara and retrieve the CDP URL.
    browser_info = instance.browser.start()
    cdp_url = browser_info.cdp_url
    print(f"Browser session started. CDP URL: {cdp_url}")
    
    # Use Playwright to connect to the browser session and log in to Outlook.
    playwright = sync_playwright().start()
    browser = playwright.chromium.connect_over_cdp(cdp_url)
    page = browser.new_page()
    
    stream_url = instance.get_stream_url().stream_url
    print(f"Instance stream URL: {stream_url}")

    page.goto("https://www.microsoft.com/en-us/microsoft-365/outlook/log-in")

            # Construct a very specific prompt instructing the agent to:
    # 1. Open a web browser.
    # 2. Navigate to the Outlook login page.
    # 3. Log in using the account DUTYtreehacks@outlook.com with the password extracted from the environment.
    # 4. Compose and send an email.
    prompt_text = f"""
Your objective is to send an email using Microsoft Outlook via its web interface on this Linux machine.
Follow these steps precisely:

 Click the "Sign in" Button.
 Log in with the account: DUTYtreehacks@outlook.com.
- When prompted, enter the password: "{outlook_password}"
 After a successful login, click on the blue "New Mail" button on the top-left corner.
 In the new email form, fill in the following details:
    - On the line of "To", type: tomyuanyucheng@gmail.com
    - One the line of "Add a Subject", type: "Hello from Scrapybara"
    - Right below "Add a Subject", type: "This is an automated email sent from Scrapybara using the account DUTYtreehacks@outlook.com."
 Finally, click Blue button immediately above "To".

Provide step-by-step automation commands or scripts to perform these actions using the available tools.
"""
    
    # Use Scrapybara's act() to process the prompt with the given tools.
    response = client.act(
        tools=[
            ComputerTool(instance),
            BashTool(instance),
            EditTool(instance)
        ],
        model=Anthropic(),
        system="You are an automated web-interaction agent that can control a Linux system to perform browser automation tasks.",
        prompt=prompt_text,
        on_step=lambda step: print(f"[Agent Step]: {step.text}")
    )
    
    # Print out the agent's response messages.
    print("\nAgent response messages:")
    for msg in response.messages:
        print(msg)
    
    # Stop the instance once done.
    instance.stop()

if __name__ == "__main__":
    send_email_via_outlook()
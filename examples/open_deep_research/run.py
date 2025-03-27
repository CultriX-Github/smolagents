#!/usr/bin/env python3

import argparse
import os
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from dotenv import load_dotenv
from huggingface_hub import login
from requests_cache import install_cache

from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
    ToolCollection,
)

from mcp import StdioServerParameters

# ------------------------ Configuration & Initialization ------------------------

# Initialize caching for web requests to improve performance
install_cache('web_cache', backend='memory', expire_after=300)  # Cache expires after 5 minutes

# Load environment variables from .env file
load_dotenv(override=True)

# Authenticate with Hugging Face Hub using the token from environment variables
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)
else:
    raise EnvironmentError("HF_TOKEN not found in environment variables.")

# Configure the SearXNG MCP server (Assuming it's running externally)
server_parameters = StdioServerParameters(
    command="",  # No command since MCP is external
    args=[],      # No args needed
    env={
        "SEARXNG_URL": "https://search.endorisk.nl",
        "SEARXNG_USERNAME": None,  # Optional
        "SEARXNG_PASSWORD": None   # Optional
    }
)

# Define authorized imports to limit agent capabilities
AUTHORIZED_IMPORTS = [
    "requests",
    "os",
    "json",
    "bs4",
    "pandas",
    "numpy",
    "torch",
    # Add more as needed, but keep it minimal for performance
]

# User-Agent string for web requests to simulate a real browser
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
)

# Browser configuration settings
BROWSER_CONFIG = {
    "viewport_size": 5120,  # Adjusted for performance
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": USER_AGENT},
        "timeout": 150,        # Reduced timeout from 300 to 150 seconds
        "max_retries": 2,      # Limit retries to prevent long wait times
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

# Ensure the downloads folder exists
os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

# Lock for thread-safe operations
append_answer_lock = threading.Lock()

# Global variable to hold the singleton agent instance
agent_instance = None

# Custom role conversions for the model
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

# ------------------------------ Argument Parsing -------------------------------

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the search agent to answer questions using web browsing tools."
    )
    parser.add_argument(
        "question",
        type=str,
        help="Example: 'How many studio albums did Mercedes Sosa release before 2007?'"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="o1",
        help="Model identifier (default: o1)"
    )
    return parser.parse_args()

# ------------------------------- Agent Creation --------------------------------

def create_agent(model_id="o1"):
    """
    Create and return a singleton agent instance.
    """
    global agent_instance
    if agent_instance is None:
        # Initialize tool collection from the MCP server
        with ToolCollection.from_mcp(server_parameters) as tool_collection:
            # Define model parameters with optimized settings
            model_params = {
                "model_id": model_id,
                "custom_role_conversions": custom_role_conversions,
                "max_completion_tokens": 4096,  # Reduced from 8192 for better performance
                "temperature": 0.7,
            }
            if model_id == "o1":
                model_params["reasoning_effort"] = "high"

            # Initialize the LLM model
            model = LiteLLMModel(**model_params)

            text_limit = 100000  # Define text inspection limit

            # Initialize the web browser with the optimized configuration
            browser = SimpleTextBrowser(**BROWSER_CONFIG)

            # Define web tools with optimized settings
            WEB_TOOLS = [
                GoogleSearchTool(provider="serper"),
                VisitTool(browser),
                PageUpTool(browser),
                PageDownTool(browser),
                FinderTool(browser),
                FindNextTool(browser),
                ArchiveSearchTool(browser),
                TextInspectorTool(model, text_limit),
            ]

            # Initialize the ToolCallingAgent with optimized parameters
            text_webbrowser_agent = ToolCallingAgent(
                model=model,
                tools=WEB_TOOLS + tool_collection.tools,  # Combine web tools with MCP tools
                max_steps=10,             # Reduced from 20
                verbosity_level=1,        # Reduced from 2
                planning_interval=4,
                name="search_agent",
                description=(
                    """A team member that will search the internet to answer your question.
                    Ask all questions that require browsing the web using complete sentences.
                    Provide as much context as possible, especially if searching within a specific timeframe.
                    """
                ),
                provide_run_summary=True,
            )

            # Enhance the agent's prompt with additional instructions
            text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += (
                """ You can navigate to .txt online files.
                If a non-HTML page is in another format, especially .pdf or a YouTube video, use the 'inspect_file_as_text' tool to inspect it.
                Additionally, if more information is needed to answer the question after some searching, use final_answer with your request for clarification as an argument."""
            )

            # Initialize the manager agent with optimized parameters
            manager_agent = CodeAgent(
                model=model,
                tools=[visualizer, TextInspectorTool(model, text_limit)],
                max_steps=12,                # Reduced from higher value
                verbosity_level=1,           # Reduced from 2
                additional_authorized_imports=AUTHORIZED_IMPORTS,
                planning_interval=4,
                managed_agents=[text_webbrowser_agent],
            )

            # Assign the manager agent to the global instance
            agent_instance = manager_agent

    return agent_instance

# ---------------------------- Asynchronous Execution ---------------------------

async def async_run_agent(agent, question):
    """
    Asynchronously run the agent with the given question.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, agent.run, question)

# ------------------------------- Main Function ---------------------------------

def main():
    """
    Main function to parse arguments, create agent, and get the answer.
    """
    args = parse_args()
    agent = create_agent(model_id=args.model_id)

    # Use ThreadPoolExecutor to run the agent concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        future = executor.submit(agent.run, args.question)
        answer = future.result()

    print(f"Got this answer: {answer}")

# ------------------------------ Entry Point ------------------------------------

if __name__ == "__main__":
    main()

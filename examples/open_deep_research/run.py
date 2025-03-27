# === run.py ===

import argparse
import os
import threading
import logging

from dotenv import load_dotenv
from huggingface_hub import login
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

# Initialize logging
logger = logging.getLogger("smolagents")
logger.setLevel(logging.DEBUG)  # Capture all levels of logs
log_handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

# Define a thread-safe list to store logs
agent_logs = []
log_lock = threading.Lock()

class ListHandler(logging.Handler):
    """
    Custom logging handler to append logs to a list.
    """
    def __init__(self, log_list):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        with log_lock:
            log_entry = self.format(record)
            self.log_list.append(log_entry)

# Attach the custom handler to capture logs
list_handler = ListHandler(agent_logs)
list_handler.setFormatter(log_formatter)
logger.addHandler(list_handler)

# Load environment variables
load_dotenv(override=True)

# Authenticate with Hugging Face Hub
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)
    logger.info("Logged into Hugging Face Hub.")
else:
    logger.warning("HF_TOKEN not found. Proceeding without authentication.")

# Define MCP server parameters with the custom endpoint
server_parameters = StdioServerParameters(
    command="",  # No command since MCP is external
    args=[],      # No args needed
    env={
        "SEARXNG_URL": "https://search.endorisk.nl",
        "SEARXNG_USERNAME": "",  # Optional
        "SEARXNG_PASSWORD": ""   # Optional
    }
)

# Define authorized imports to limit agent capabilities
AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
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

# Define custom role conversions for the model
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
    Create and return an agent instance configured to use the custom MCP server.
    """
    # Initialize tool collection from the custom MCP server
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
        logger.info(f"Initialized LiteLLMModel with model_id={model_id}")

        text_limit = 100000  # Define text inspection limit

        # Initialize the web browser with the optimized configuration
        browser = SimpleTextBrowser(**BROWSER_CONFIG)
        logger.info("Initialized SimpleTextBrowser with custom configuration.")

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
        logger.info("Initialized web tools for ToolCallingAgent.")

        # Initialize the ToolCallingAgent with optimized parameters
        text_webbrowser_agent = ToolCallingAgent(
            model=model,
            tools=WEB_TOOLS + tool_collection.tools,  # Combine web tools with MCP tools
            max_steps=10,             # Reduced from 20
            verbosity_level=2,        # Set to 2 for detailed logs
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
        logger.info("Initialized ToolCallingAgent.")

        # Enhance the agent's prompt with additional instructions
        text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += (
            """ You can navigate to .txt online files.
            If a non-HTML page is in another format, especially .pdf or a YouTube video, use the 'inspect_file_as_text' tool to inspect it.
            Additionally, if more information is needed to answer the question after some searching, use `final_answer` with your request for clarification as an argument."""
        )
        logger.debug("Enhanced Agent prompt with additional instructions.")

        # Initialize the manager agent with optimized parameters
        manager_agent = CodeAgent(
            model=model,
            tools=[visualizer, TextInspectorTool(model, text_limit)],
            max_steps=12,                # Reduced from higher value
            verbosity_level=2,           # Detailed logs
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            planning_interval=4,
            managed_agents=[text_webbrowser_agent],
        )
        logger.info("Initialized Manager CodeAgent.")

        return manager_agent

# ------------------------------- Main Function ---------------------------------

def main():
    args = parse_args()
    logger.info(f"Received question: {args.question} with model_id={args.model_id}")

    agent = create_agent(model_id=args.model_id)

    answer = agent.run(args.question)

    print(f"Got this answer: {answer}")
    logger.info("Agent has completed processing the question.")

if __name__ == "__main__":
    main()

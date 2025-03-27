import argparse
import os
import threading
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests_cache

# Initialize caching
requests_cache.install_cache('web_cache', backend='memory', expire_after=300)

# Configure the SearXNG MCP server (assume it's running externally)
server_parameters = StdioServerParameters(
    command="",  # No command since MCP is external
    args=[],  # No args needed
    env={
        "SEARXNG_URL": "https://search.endorisk.nl",
        "SEARXNG_USERNAME": None,
        "SEARXNG_PASSWORD": None
    }
)

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

load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, help="e.g., 'How many studio albums did Mercedes Sosa release before 2007?'"
    )
    parser.add_argument("--model-id", type=str, default="o1")
    return parser.parse_args()

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
BROWSER_CONFIG = {
    "viewport_size": 5120,  # Simplified
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 150,
        "max_retries": 2,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

agent_instance = None  # Global agent instance

def create_agent(model_id="o1"):
    global agent_instance
    if agent_instance is None:
        with ToolCollection.from_mcp(server_parameters) as tool_collection:
            model_params = {
                "model_id": model_id,
                "custom_role_conversions": custom_role_conversions,
                "max_completion_tokens": 4096,  # Reduced
                "temperature": 0.7,
            }
            if model_id == "o1":
                model_params["reasoning_effort"] = "high"
            model = LiteLLMModel(**model_params)
            text_limit = 100000
            browser = SimpleTextBrowser(**BROWSER_CONFIG)
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
            text_webbrowser_agent = ToolCallingAgent(
                model=model,
                tools=WEB_TOOLS + tool_collection.tools,  # Combined tools
                max_steps=10,  # Reduced
                verbosity_level=1,  # Reduced
                planning_interval=4,
                name="search_agent",
                description="""A team member that will search the internet to answer your question.
        Ask him for all your questions that require browsing the web.
        Provide him as much context as possible, especially if you need to search within a specific timeframe.
        Use complete sentences for requests, e.g., "Find me this information (...)" rather than just keywords.""",
                provide_run_summary=True,
            )
            text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
            If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
            Additionally, if after some searching you need more information to answer the question, use final_answer with your request for clarification as an argument to request more information."""
            manager_agent = CodeAgent(
                model=model,
                tools=[visualizer, TextInspectorTool(model, text_limit)],
                max_steps=12,
                verbosity_level=1,
                additional_authorized_imports=AUTHORIZED_IMPORTS,
                planning_interval=4,
                managed_agents=[text_webbrowser_agent],
            )
            agent_instance = manager_agent
    return agent_instance

async def async_run_agent(agent, question):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, agent.run, question)

def main():
    args = parse_args()
    agent = create_agent(model_id=args.model_id)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future = executor.submit(agent.run, args.question)
        answer = future.result()
    
    print(f"Got this answer: {answer}")

if __name__ == "__main__":
    main()

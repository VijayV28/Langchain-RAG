from dotenv import load_dotenv
from langchain_community.utilities import ApifyWrapper
from langchain_core.documents import Document

# Loading the API keys
load_dotenv("../.env")

# Initializing the website crawler
apify = ApifyWrapper()
loader = apify.call_actor(
    actor_id="apify/website-content-crawler", run_input={"startUrls": [{"url": {}}]}
)

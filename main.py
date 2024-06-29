from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from agents.Agents import NewsAgents  # Update this
from agents.Utils import print_agent_output, GROQ_LLM  # Update this
from agents.Main import NEWSAGENCY

import sys
sys.path.append('./agents')
import Main as module_main

class News(BaseModel):
    headline: str

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/headline")
async def create_news(news: News):
    news_headline = news.headline

    await module_main.passHeadline(news_headline)
    return {
        "results": results,
        "usage_metrics": usage_metrics
    }

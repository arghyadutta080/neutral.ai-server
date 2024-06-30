from agents.Utils import print_agent_output, GROQ_LLM
import os
import json
from dotenv import load_dotenv
from crewai import Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from agents.Utils import print_agent_output, GROQ_LLM
from langchain_groq import ChatGroq
from typing import Union, List, Tuple, Dict
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)


class NEWSAGENCY:
    def __init__(self):

        self.groq_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192"
        )

        from Agents import NewsAgents
        from Tasks import NewsAgencyTasks

        self.agents = NewsAgents()
        self.tasks = NewsAgencyTasks()
        self.search_tool = SerperDevTool()

        self.crew = None

    def run(self, headline: str):
        # agents
        # news_scraper_agent = self.agents.make_news_scraper_agent()
        # event_detection_agent = self.agents.make_event_detection_agent()
        # fact_checking_agent = self.agents.make_fact_checking_agent()
        content_generation_agent = self.agents.make_content_generation_agent()
        editor_agent = self.agents.make_editor_agent()
        S_news_scraper_agent = self.agents.search_news_scraper_agent()
        S_news_fact_checker_agent = self.agents.Search_make_fact_checking_agent()

        # tasks
        # scraping_task = self.tasks.scrape_news()
        # detection_task = self.tasks.detect_events()
        # fact_checking_task = self.tasks.fact_check_events()
        content_generation_task = self.tasks.generate_news_content(headline)
        editing_task = self.tasks.edit_news_content(headline)
        headline_scrape_task = self.tasks.headline_scrape_news(headline)
        s_fact_chack_task = self.tasks.S_fact_check_events(headline)

        # crew with agents and tasks
        self.crew = Crew(
            agents=[S_news_scraper_agent, S_news_fact_checker_agent,
                    content_generation_agent],
            tasks=[headline_scrape_task, s_fact_chack_task,
                   content_generation_task],
            verbose=2,
            process=Process.sequential,
            full_output=True,
            share_crew=False,
            step_callback=lambda x: print_agent_output(x, "MasterCrew Agent")
        )

        results = self.crew.kickoff()

        return results, self.crew.usage_metrics


async def passHeadline(headline: str):
    news_automation = NEWSAGENCY()
    results, usage_metrics = news_automation.run(headline)
    print("Crew Work Results:")
    print(results)
    print('------------------------------------------------------------------')
    print("Usage Metrics:")
    print(usage_metrics)

    with open("news_articles.json") as file:
        data = file.read()

    return { 'response': data }


if __name__ == "__main__":
    passHeadline(headline)

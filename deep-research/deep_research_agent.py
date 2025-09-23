from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch

from dotenv import load_dotenv
from prompt import (
    INDEPENDENT_AGENT_PROMPT,
    WORKER_PROMPT,
    QUERY_SPLITTER_PROMPT,
    UNIQUE_RESEARCH_TOPIC_PROMPT,
)

import json
import time

WEB_SEARCH_RELEVANCE_SCORE_THRESHOLD = 0.30
MODEL_NAME = "gemini-2.5-pro"


class DeepResearchSharedState(TypedDict):
    question: str
    model: ChatGoogleGenerativeAI
    depth: int
    breadth: int
    citations: list[str]
    research_topics: list[str]
    reports: list[str]


class IndependentAgentOutput(BaseModel):
    sub_queries: list[str] = Field(
        description="List of sub-queries to be answered independently"
    )


class WorkerOutput(BaseModel):
    answer: str = Field(description="The answer to the question")


class QuerySplitterOutput(BaseModel):
    can_split: bool = Field(
        description="Whether the query can be split into sub-queries"
    )


class UniqueResearchTopicOutput(BaseModel):
    is_unique: bool = Field(description="Whether the research topic is unique")


class WebSearchResult(TypedDict):
    cited_url: str
    content: str
    score: float


## -------- Helper Methods Start-------- ##
def build_model():
    """Builds the LLM"""
    return ChatGoogleGenerativeAI(model=MODEL_NAME)


def sleep_thread():
    """Sleep for a few seconds to avoid rate limits"""
    sleep_time = 1
    print(f"Sleeping for {sleep_time} seconds to avoid rate limits...")
    time.sleep(sleep_time)
    print(f"Woke Up!")


def can_split_into_subtasks(query: str, model: ChatGoogleGenerativeAI) -> bool:
    """Determine if the query can be split into sub-tasks"""
    model_with_structured_output = model.with_structured_output(QuerySplitterOutput)
    query_splitter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_SPLITTER_PROMPT),
            ("human", "Query: \n\n {query}"),
        ]
    )

    query_splitter_retriever = query_splitter_prompt | model_with_structured_output
    query_splitter_response: QuerySplitterOutput = query_splitter_retriever.invoke(
        {"query": query}
    )
    sleep_thread()

    return query_splitter_response.can_split


def is_different_research_topic(
    research_topic: str, previous_research_topics: list, model: ChatGoogleGenerativeAI
) -> bool:
    """Determine if the research topic is different from previous research topics"""
    model_with_structured_output = model.with_structured_output(
        UniqueResearchTopicOutput
    )
    unique_research_topic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", UNIQUE_RESEARCH_TOPIC_PROMPT),
            (
                "human",
                "Research Topic: \n\n {research_topic} \n\n Previous Research Topics: \n\n {previous_research_topics}",
            ),
        ]
    )

    unique_research_topic_retriever = (
        unique_research_topic_prompt | model_with_structured_output
    )
    unique_research_topic_response: UniqueResearchTopicOutput = (
        unique_research_topic_retriever.invoke(
            {
                "research_topic": research_topic,
                "previous_research_topics": previous_research_topics,
            }
        )
    )

    return unique_research_topic_response.is_unique


def parse_web_search_results(web_search_result):
    """Parse the web search results and filter based on relevance score"""
    search_results = []

    if not web_search_result or "results" not in web_search_result:
        return search_results

    for result in web_search_result["results"]:
        print(f"Web Search Result: {result}")
        cited_url = result["url"]
        search_content = result["content"]
        search_score = result["score"]

        if search_score >= WEB_SEARCH_RELEVANCE_SCORE_THRESHOLD:
            search_results.append(
                WebSearchResult(
                    cited_url=cited_url, content=search_content, score=search_score
                )
            )

    return search_results


def web_search(query):
    """Perform a web search and return the results"""
    print(f"\n Performing Web Search for {query} \n")

    web_search_tool = TavilySearch(max_results=5)
    web_results = web_search_tool.invoke({"query": query})

    return parse_web_search_results(web_results)


def build_deep_research_report(shared_agent_state):
    """Build the final research report from the shared agent state"""
    research_report = "# Deep Research Report\n\n"
    research_report += "## Table of Contents \n"
    for topic in shared_agent_state["research_topics"]:
        research_report += f"- {topic}\n"

    research_report += "\n## Report \n"
    for report in shared_agent_state["reports"]:
        research_report += report + "\n\n"

    research_report += "\n## Citations\n"
    unique_citations = set(shared_agent_state["citations"])
    for citation in unique_citations:
        research_report += f"- {citation} \n"

    return research_report


## -------- Helper Methods End -------- ##


def initiate_deep_research(query: str, depth: int = 2, breadth: int = 2):
    """Initiate the deep research agent with the given query, depth, and breadth"""
    model = build_model()
    shared_agent_state: DeepResearchSharedState = {
        "question": query,
        "depth": depth,
        "breadth": breadth,
        "model": model,
        "citations": [],
        "research_topics": [],
        "reports": [],
    }

    agent_stats_response: DeepResearchSharedState = supervisor(shared_agent_state)
    return build_deep_research_report(agent_stats_response)


def supervisor(agent_state: DeepResearchSharedState):
    """The Supervisor agent that decides whether to split the query or delegate to a worker"""
    query = agent_state["question"]
    model = agent_state["model"]
    current_depth = agent_state["depth"]
    print(f"Initiating Supervisor for '{query}'")

    can_split = False
    if current_depth > 0:
        can_split = can_split_into_subtasks(query, model)
        print(f"Should agent split the query? => {can_split}")
    else:
        print(f"Depth Limit of {current_depth} reached")

    if can_split:
        agent_state_response: DeepResearchSharedState = independent_agent(agent_state)
    else:
        is_unique_topic = is_different_research_topic(
            agent_state["question"],
            agent_state["research_topics"],
            agent_state["model"],
        )

        if is_unique_topic:
            agent_state_response: DeepResearchSharedState = worker(agent_state)
        else:
            print(
                f"The research topic {agent_state['question']} is not unique and hence Skipping this!"
            )

    return agent_state_response


def independent_agent(agent_state: DeepResearchSharedState):
    """The Independent Agent that splits the query into sub-queries and delegates to Supervisors"""
    query = agent_state["question"]
    model = agent_state["model"]
    print(f"Initiating Independent Agent for '{query}'")

    model_with_structured_output = model.with_structured_output(IndependentAgentOutput)
    independent_agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INDEPENDENT_AGENT_PROMPT),
            ("human", "Query: \n\n {query} \n\n, Limit: {limit}"),
        ]
    )

    independent_agent_retriever = (
        independent_agent_prompt | model_with_structured_output
    )

    independent_sub_tasks: IndependentAgentOutput = independent_agent_retriever.invoke(
        {"query": query, "limit": agent_state["breadth"]}
    )
    sleep_thread()

    research_reports = agent_state["reports"]
    current_citations = agent_state["citations"]
    current_research_topics = agent_state["research_topics"]

    current_depth: int = agent_state["depth"]
    current_breadth: int = agent_state["breadth"]
    supervisor_responses = []

    print(f"Sub-queries spawned: {independent_sub_tasks.sub_queries}")
    for sub_query in independent_sub_tasks.sub_queries:
        supervisor_agent_state: DeepResearchSharedState = {
            "question": sub_query,
            "depth": current_depth - 1,  # Reducing the depth for each sub-task
            "breadth": current_breadth - 2,  # Reducing the breadth for each sub-task
            "model": agent_state["model"],
            "citations": [],
            "research_topics": [],
            "reports": [],
        }
        supervisor_response = supervisor(supervisor_agent_state)
        supervisor_responses.append(supervisor_response)

    research_reports.append(f"## {query}\n\n")
    # Fan-in all the responses from Supervisors
    for supervisor_response in supervisor_responses:
        research_reports.extend(supervisor_response["reports"])
        current_citations.extend(supervisor_response["citations"])
        current_research_topics.extend(supervisor_response["research_topics"])

    agent_state["reports"] = research_reports
    agent_state["citations"] = current_citations
    agent_state["research_topics"] = current_research_topics

    return agent_state


def worker(agent_state: DeepResearchSharedState):
    """A straight-forward worker that just answers the query"""

    query = agent_state["question"]
    model = agent_state["model"]

    print(f"Researching for the Query => '{query}'")
    web_search_results = web_search(query)
    print("Web Search completed!")

    if web_search_results is None or len(web_search_results) == 0:
        print("No relevant web search results found. Skipping this research topic!")
        return agent_state

    model_with_structured_output = model.with_structured_output(WorkerOutput)
    worker_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WORKER_PROMPT),
            (
                "human",
                "Query: \n\n {query} \n\n, Web Search Results: {web_search_results}",
            ),
        ]
    )

    worker_retriever = worker_prompt | model_with_structured_output
    print("Querying the LLM...")
    worker_result: WorkerOutput = worker_retriever.invoke(
        {"query": query, "web_search_results": web_search_results}
    )

    if (
        worker_result is None
        or worker_result.answer is None
        or worker_result.answer.strip() == ""
    ):
        print("Research Topic Skipped!")
        return agent_state

    agent_state["research_topics"].append(query)
    agent_state["citations"].extend(
        [web_search_result["cited_url"] for web_search_result in web_search_results]
    )
    agent_state["reports"].append(" \n ### " + query + "\n\n" + worker_result.answer)
    sleep_thread()
    print("Research Done!")

    return agent_state


def append_to_model_output_file(task_id: int, prompt: str, research_report: str):
    """Append the model output to a JSONL file"""
    filename = "research_output/model_output/output.jsonl"
    model_output = {"id": task_id, "prompt": prompt, "article": research_report}

    try:
        with open(filename, "r") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []  # Initialize as an empty list if the file doesn't exist

    # Example for appending to a list within a dictionary
    if isinstance(existing_data, list):
        existing_data.append(model_output)

    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)


## Runner Method ##
def start_deep_research(task):
    """Entry point to start the deep research agent"""
    load_dotenv()  # Load environment variables from .env file
    task_id = task["id"]
    file_name = f"output_final_report_{task_id}"
    query = task["prompt"]
    response = initiate_deep_research(query, depth=2, breadth=5)

    append_to_model_output_file(task_id, query, response)
    with open(f"research_output/{file_name}.md", "w") as file_object:
        # Write the string to the file
        file_object.write(response)


task = {
    "id": 62,
    "topic": "Science & Technology",
    "language": "en",
    "prompt": "What are the most effective approaches to scaling ion trap quantum computing from small-scale demonstration projects to large-scale systems capable of solving real-world problems? This research should investigate the various proposed scaling strategies, assess their feasibility, and evaluate which approaches are most likely to succeed based on current technological advancements and practical implementation challenges.",
}


start_deep_research(task)

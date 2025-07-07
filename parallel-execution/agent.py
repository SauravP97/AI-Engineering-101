from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from langgraph.graph import START, StateGraph, END

BLOOM_FILTER_ARTICLE_LINK = "bloom-filters.txt"
GRAPH_DB_ARTICLE_LINK = "graph-db.txt"

# To Handle: https://langchain-ai.github.io/langgraph/troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE/
def custom_reducer(obj1: str, obj2: str):
  return obj1 if obj1.strip() else obj2


class SharedState(TypedDict):
  """
  Represents the shared state of our graph.
  """
  bloom_filter_article_link: Annotated[str, custom_reducer]
  graph_db_article_link: Annotated[str, custom_reducer]

  bloom_filter_article_summary: Annotated[str, custom_reducer]
  graph_db_article_summary: Annotated[str, custom_reducer]

  summary: Annotated[str, custom_reducer]


def get_article_content(article_name: str) -> str:
    """ Function to get the content of an article."""
    profile = open('./articles/' + article_name, 'r')
    profile_content = profile.read()
    return profile_content


def get_bloom_filter_article_summary(state: SharedState) -> SharedState:
    """ Function to get the summary of the Bloom Filter article."""
    print('Summarizing Bloom Filter article...')
    article_content = get_article_content(state['bloom_filter_article_link'])

    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    response = llm.invoke(
        f"Summarize the following article:\n\n{article_content}\n\n"
        "Please provide a concise summary that captures the main points. Please keep the summary under 200 words."
    )
    state['bloom_filter_article_summary'] = response.content.strip() if response else "No summary available."

    return state


def get_graph_db_article_summary(state: SharedState) -> SharedState:
    """ Function to get the summary of the Graph database article."""
    print('Summarizing Graph DB article...')
    article_content = get_article_content(state['graph_db_article_link'])

    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    response = llm.invoke(
        f"Summarize the following article:\n\n{article_content}\n\n"
        "Please provide a concise summary that captures the main points. Please keep the summary under 200 words."
    )
    state['graph_db_article_summary'] = response.content.strip() if response else "No summary available."

    return state


def join_node(state: SharedState) -> SharedState:
    """
    This node acts as a router/conditional check.
    It decides the next step based on whether A and B have completed.
    """
    print(f"Bloom Filter article summary fetched = {state.get('bloom_filter_article_summary') is not None}")
    print(f"Graph DB article summary fetched = {state.get('graph_db_article_summary') is not None}")

    if state.get("bloom_filter_article_summary") is not None and state.get("graph_db_article_summary") is not None:
        print("Articles summarized. Proceeding to final processing.")
        # Perform the work that requires both A and B's output
        summarized_result = f"Bloom Filter Article summary: \n\n {state['bloom_filter_article_summary']} \n\n\n"
        summarized_result += f"Graph DB Article summary: \n\n {state['graph_db_article_summary']} \n\n\n"
        state["summary"] = summarized_result

        return state

    print("Still waiting for one or more nodes...")
    return state


def build_graph():
  load_dotenv()

  # Building a Graph
  # State of the Graph that will be shared among nodes.
  workflow = StateGraph(SharedState)

  # Add nodes.
  workflow.add_node("get_bloom_filter_article_summary", get_bloom_filter_article_summary)
  workflow.add_node("get_graph_db_article_summary", get_graph_db_article_summary)
  workflow.add_node("join_node", join_node)

  # Define the edges of the graph.
  workflow.add_edge(START, "get_bloom_filter_article_summary")
  workflow.add_edge(START, "get_graph_db_article_summary")
  workflow.add_edge("get_bloom_filter_article_summary", "join_node")
  workflow.add_edge("get_graph_db_article_summary", "join_node")
  workflow.add_conditional_edges(
    "join_node",
    # This function determines the next node based on the state
    lambda state: END if state.get("summary") is not None else "join_node"
  )

  graph = workflow.compile()

  response = graph.invoke({
      'bloom_filter_article_link': BLOOM_FILTER_ARTICLE_LINK,
      'graph_db_article_link': GRAPH_DB_ARTICLE_LINK,
  })

  # print(graph.get_graph().draw_mermaid())

  return response

agent_response = build_graph()
print(f'\n\n\n Summary generated: \n\n {agent_response["summary"]}')
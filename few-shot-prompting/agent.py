import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated
from langgraph.graph import START, StateGraph, END

from dotenv import load_dotenv

MAX_FEW_SHOT_EXAMPLES = 3

BASE_PROMPT = '''You are Saurav Prateek. 
You post highlighting information related to technical concepts on Linkedin.
Your field of interests are Generative AI and Software Engineering.
You use emojis. You use exclamation points but are not overly enthusiastic. 
You are not overly formal.
You are not "salesy". You are nice.

When given an article, write a summarized linkedin post about it. 
Make it relevant and specific to the article at hand.

Pay attention to the examples below. These are good examples. 
Generate future summarized posts in the style of the posts below.
'''

class SharedState(TypedDict):
    post_url: str
    post_content: str
    few_shot_examples: list
    linkedin_post_content: str


def get_article_content_for_post(shared_state: SharedState):
    post_content = get_content_from_url(shared_state['post_url'])
    shared_state['post_content'] = post_content
    return shared_state


def prepare_few_shot_data(shared_state: SharedState):
    print("Running prepare_few_shot_data...")
    linkedin_posts = get_linkedin_posts()
    few_shot_examples = []

    for linkedin_post in linkedin_posts:
        newsletter_article_url = linkedin_post['url']
        newsletter_article_content = get_content_from_url(newsletter_article_url)
        newsletter_linkedin_post = linkedin_post['post_content']
        
        few_shot_examples.append({
            'article_content': newsletter_article_content,
            'post_content': newsletter_linkedin_post
        })
    
    shared_state['few_shot_examples'] = few_shot_examples[:MAX_FEW_SHOT_EXAMPLES]
    return shared_state


def summarize_linkedin_post(shared_state: SharedState):
    print("Summarizing Linkedin Post for you...")
    messages = [SystemMessage(content=BASE_PROMPT)]

    for few_shot_example in shared_state['few_shot_examples']:
        messages.append(HumanMessage(content=few_shot_example['article_content']))
        messages.append(AIMessage(content=few_shot_example['post_content']))
    
    messages.append(HumanMessage(content=shared_state['post_content']))

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    response = model.invoke(messages)

    shared_state['linkedin_post_content'] = response.content.strip() if response else "No summary available."

    return shared_state


def get_linkedin_posts():
    linkedin_posts = open('./dataset/linkedin-posts.json', 'r')
    post_json_data = linkedin_posts.read()
    return json.loads(post_json_data)['examples']


def get_content_from_url(url:str):
    documents = WebBaseLoader(url).load()
    page_content = ''

    for document in documents:
        page_content += document.page_content
    
    return page_content.strip()


def build_graph():
    load_dotenv()
    # Building a Graph
    # State of the Graph that will be shared among nodes.
    workflow = StateGraph(SharedState)

    workflow.add_node("get_article_content_for_post", get_article_content_for_post)
    workflow.add_node("prepare_few_shot_data", prepare_few_shot_data)
    workflow.add_node("summarize_linkedin_post", summarize_linkedin_post)

    workflow.add_edge(START, "get_article_content_for_post",)
    workflow.add_edge("get_article_content_for_post", "prepare_few_shot_data")
    workflow.add_edge("prepare_few_shot_data", "summarize_linkedin_post")
    workflow.add_edge("summarize_linkedin_post", END)

    graph = workflow.compile()

    response = graph.invoke({
        "post_url": "https://www.linkedin.com/pulse/parallel-execution-nodes-langgraph-enhancing-your-graph-prateek-qqwrc/"
    })

    return response


state = build_graph()
print("Summarized Linkedin Post: \n")
print(state['linkedin_post_content'])
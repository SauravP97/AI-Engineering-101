from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from langgraph.graph import START, StateGraph, END

CANDIDATE_PROFILE = 'profile2.txt'
RECEIVER_PROFILE = 'profile1.txt'


def custom_reducer(obj1, obj2):
  return obj2


class SharedState(TypedDict):
  """
  Represents the state of our graph.
  """
  candidate_profile: Annotated[str, custom_reducer]
  receiver_profile: Annotated[str, custom_reducer]

  candidate_profile_content: Annotated[str, custom_reducer]
  receiver_profile_content: Annotated[str, custom_reducer]
  
  candidate_profile_information: Annotated[str, custom_reducer]
  receiver_profile_information: Annotated[str, custom_reducer]

  pitch: Annotated[str, custom_reducer]


def get_profile_content(profile) -> str:
    """ Get Profile data from Linkedin profile url or from a Profile file."""
    profile = open('./profiles/' + profile, 'r')
    profile_content = profile.read()
    return profile_content


def get_candidate_profile_content(shared_state: SharedState) -> str:
    """ Get Candidate Profile data from Linkedin profile url or from a Profile file."""
    shared_state['candidate_profile_content'] = get_profile_content(shared_state['candidate_profile'])
    return shared_state


def get_receiver_profile_content(shared_state: SharedState) -> str:
    """ Get Receiver Profile data from Linkedin profile url or from a Profile file."""
    shared_state['receiver_profile_content'] = get_profile_content(shared_state['receiver_profile'])
    return shared_state


def extract_receiver_profile_information(shared_state: SharedState) -> str:
    """Extract useful information from the profile content."""

    query = "Get Profile details like name, organization and current role from the profile content."
    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke([
            {"role": "system", "content": "You are a helpful assistant that extracts profile details from the provided content."},
            {"role": "user", "content": f"Profile URL: {shared_state['receiver_profile_content']} \n\n Question: {query}"}
        ])
    
    shared_state['receiver_profile_information'] = response.content
    return shared_state


def extract_candidate_profile_information(shared_state: SharedState) -> str:
    """Extract useful information from the candidate profile content."""

    query = "Get Profile details like name, organization, current role, experience and skill from the profile content."
    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke([
            {"role": "system", "content": "You are a helpful assistant that extracts profile details from the provided content."},
            {"role": "user", "content": f"Profile URL: {shared_state['candidate_profile_content']} \n\n Question: {query}"}
        ])
    
    shared_state['candidate_profile_information'] = response.content
    return shared_state


def write_a_referral_pitch(shared_state: SharedState):
    """ Write a referral pitch based on the receiver's and candidate profile information."""
    query = '''
    You are a candidate applying for an AI Engineer role.
    Write a referral pitch for applying to an open position in the receiver's organization 
    based on the receiver's profile information and the candidate profile information. 
    The receiver profile information and candidate profile information is provided to you and 
    the receiver information has the receiver name, organization, and current role.
    The candidate profile information has the candidate name, organization, current role, experience, and skills.
    
    The pitch should be concise, professional, and highlight the candidate's skills and
    experiences that make them a good fit for the position.

    Keep the pitch message concise and to the point, under 100 words.
    '''
    print("Writing Referral Pitch...")
    print(f'Receiver Profile info {shared_state["receiver_profile_information"]}')
    print(f'Candidate Profile info {shared_state["candidate_profile_information"]}')

    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke([
            {
                "role": "system", 
                "content": f"""You are the Candidate trying to write a referral pitch for applying to 
                an open position in the receiver's organization.
                """
            },
            {
                "role": "user", 
                "content": f"""
                Receiver Information: {shared_state['receiver_profile_information']} 
                \n\n Candidate Information: {shared_state['candidate_profile_information']} 
                \n\n Question: {query}
                """
            }
        ])
    
    shared_state['pitch'] = response.content
    return shared_state


def build_graph():
  load_dotenv()

  # Building a Graph
  # State of the Graph that will be shared among nodes.
  workflow = StateGraph(SharedState)

  # Add nodes.
  workflow.add_node("get_candidate_profile_content", get_candidate_profile_content)
  workflow.add_node("get_receiver_profile_content", get_receiver_profile_content)
  workflow.add_node("extract_candidate_profile_information", extract_candidate_profile_information)
  workflow.add_node("extract_receiver_profile_information", extract_receiver_profile_information)
  workflow.add_node("write_a_referral_pitch", write_a_referral_pitch)

  workflow.add_edge(START, "get_candidate_profile_content")
  workflow.add_edge("get_candidate_profile_content", "get_receiver_profile_content")
  workflow.add_edge("get_receiver_profile_content", "extract_candidate_profile_information")
  workflow.add_edge("extract_candidate_profile_information", "extract_receiver_profile_information")
  workflow.add_edge("extract_receiver_profile_information", "write_a_referral_pitch")
  workflow.add_edge("write_a_referral_pitch", END)

  graph = workflow.compile()

  response = graph.invoke({
      'candidate_profile': CANDIDATE_PROFILE,
      'receiver_profile': RECEIVER_PROFILE,
  })

  # print(graph.get_graph().draw_mermaid())

  return response


agent_response = build_graph()
print(f'Referral Pitch {agent_response["pitch"]}')
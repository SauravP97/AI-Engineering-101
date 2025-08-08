from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START

from prompts import DOCUMENT_GRADER_PROMPT, HALLUCINATION_GRADER_PROMPT, ANSWER_GRADER_PROMPT


KNOWLEDGE_BASE_URLS = [
    "https://www.linkedin.com/pulse/parallel-execution-nodes-langgraph-enhancing-your-graph-prateek-qqwrc/",
    "https://www.linkedin.com/pulse/tool-calling-langchain-do-more-your-ai-agents-saurav-prateek-so20c",
]

# Shared Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        model: LLM model used for generation
        vector_store: vector store for RAG
        hallucinated: whether the generation is grounded in documents
        valid_answer: whether the generation answers the question
    """

    question: str
    generation: str
    documents: List[str]
    model: ChatOpenAI
    vector_store: Chroma
    hallucinated: bool
    valid_answer: bool

# Data model for grading documents
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Data model for grading hallucinations
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Data model for grading the final answer
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def create_model(state):
    print("---CREATE GPT MODEL---")
    state['model'] = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return state


def build_vector_store(state):
    print("---BUILD VECTOR STORE---")
    docs = [WebBaseLoader(url).load() for url in KNOWLEDGE_BASE_URLS]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vector_store = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    state['vector_store'] = vector_store.as_retriever()

    return state


def get_relevant_documents(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE RELEVANT DOCUMENTS---")
    question = state["question"]

    # Retrieval
    vector_store = state["vector_store"]
    documents = vector_store.get_relevant_documents(question)
    state["documents"] = documents

    return state


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    structured_llm_grader = state["model"].with_structured_output(GradeDocuments)
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DOCUMENT_GRADER_PROMPT),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    
    state['documents'] = filtered_docs
    return state


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, END ---"
        )
        return "end"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "continue"


def generate_answer(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    prompt = hub.pull("rlm/rag-prompt")

    # RAG generation
    rag_chain = prompt | state['model'] | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    state['generation'] = generation

    return state


def check_for_hallucination(state):
    """
    Determines whether the LLM hallucinated or not.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call depending on the hallucination check
    """

    print("---CHECK HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]

    structured_llm_grader = state['model'].with_structured_output(GradeHallucinations)
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", HALLUCINATION_GRADER_PROMPT),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    hallucination_grader = hallucination_prompt | structured_llm_grader
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: [NO HALLUCINATIONS] GENERATION IS GROUNDED IN DOCUMENTS---")
        state['hallucinated'] = False
    else:
        print("---DECISION: [MODEL HALLUCINATED] GENERATION IS NOT GROUNDED IN DOCUMENTS")
        state['hallucinated'] = True
    
    return state


def grade_answer(state):
    """
    Determines whether the LLM generated relevant answer.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call depending on the answer grader check
    """

    print("---CHECK GENERATED ANSWER RELEVANCE---")
    question = state["question"]
    generation = state["generation"]

    structured_llm_grader = state['model'].with_structured_output(GradeAnswer)
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_GRADER_PROMPT),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_grader
    score = answer_grader.invoke(
        {"question": question, "generation": generation}
    )
    grade = score.binary_score

    # Check generated answer relevance
    if grade == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        state['valid_answer'] = True
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION")
        state['valid_answer'] = False
    
    return state


def build_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("create_model", create_model)
    workflow.add_node("build_vector_store", build_vector_store)
    workflow.add_node("get_relevant_documents", get_relevant_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("check_for_hallucination", check_for_hallucination)
    workflow.add_node("grade_answer", grade_answer)

    # Build graph
    workflow.add_edge(START, "create_model")
    workflow.add_edge("create_model", "build_vector_store")
    workflow.add_edge("build_vector_store", "get_relevant_documents")
    workflow.add_edge("get_relevant_documents", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "continue": "generate_answer",
            "end": END,
        },
    )
    workflow.add_edge("generate_answer", "check_for_hallucination")
    workflow.add_edge("check_for_hallucination", "grade_answer")

    # Compile
    return workflow.compile()


load_dotenv()

graph = build_graph()
# print(graph.get_graph().draw_mermaid())

response = graph.invoke({
    "question": "What is a Disjoint Set data structure?"
})

print("---FINAL RESPONSE---")

if 'hallucinated' in response.keys() and response['hallucinated']:
    print("Model Hallucinated, generation is not grounded in documents. \n")

if 'valid_answer' in response.keys() and not response['valid_answer']:
    print("Answer is not valid for the question. \n")

if 'generation' in response.keys():
    print("Generated Answer: \n")
    print(response['generation'])
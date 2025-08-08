DOCUMENT_GRADER_PROMPT = """
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

HALLUCINATION_GRADER_PROMPT = """
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 

    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

ANSWER_GRADER_PROMPT = """
    You are a grader assessing whether an answer addresses / resolves a question \n 
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
"""
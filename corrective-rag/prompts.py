
GRADE_DOCUMENTS_PROMPT = """
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

QUESTION_REWRITER_PROMPT = """
    You a question re-writer that converts an input question to a better version that is optimized \n 
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
"""
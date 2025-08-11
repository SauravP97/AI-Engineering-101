from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

from prompts import RESUME_ANALYSIS_PROMPT, RELATABLE_JOB_ROLES_PROMPT

import requests


KNOWLEDGE_BASE_URLS = [
    'https://www.google.com/about/careers/applications/jobs/results?q=%22software%20engineer%22&employment_type=FULL_TIME&company=Google&location=India&page=10'
]

BASE_URL = 'https://www.google.com/about/careers/applications/jobs/results?q=%22software%20engineer%22&employment_type=FULL_TIME&company=Google&location=India'

GOOGLE_JOB_LINK_PREFIX = 'https://www.google.com/about/careers/applications/'
FETCHED_GOOGLE_JOB_URL_PREFIX = 'jobs/results/'

MINIMUM_QUALIFICATION_QUESTION = 'What are the Minimum Qualifications in the job?'
PREFERRED_QUALIFICATION_QUESTION = 'What are the Preferred Qualifications in the job?'
JOB_RESPONSIBILITIES_QUESTION = 'What are the Responsibilities in the job?'

MAX_ROLES = 3

########## Helper Methods Start ##########

def get_all_urls(base_url, page=1):
    try:
        # Fetch the HTML content from the URL
        response = requests.get(base_url + f'&page={page}' + f'&target_level=MID')
        response.raise_for_status()  # This will raise an exception for bad responses (4xx or 5xx)

        # Parse the HTML content
        soup = Soup(response.text, 'html.parser')

        # Find all 'a' tags (anchor tags) which contain links
        links = soup.find_all('a')

        # Store the URLs in a list
        all_urls = []
        for link in links:
            href = link.get('href')  # Get the value of the 'href' attribute
            if href:
                all_urls.append(href)

        return all_urls

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []


def get_google_job_urls(urls):
    google_job_urls = []
    
    for url in urls:
        if url.startswith(FETCHED_GOOGLE_JOB_URL_PREFIX):
            google_job_urls.append(GOOGLE_JOB_LINK_PREFIX + url)
    
    return google_job_urls


def get_url_content(urls):
    return WebBaseLoader(urls, requests_per_second=10).load()

########## Helper Methods End ##########

def get_google_jobs_content():
    page = 1
    google_job_urls = []

    while(True):
        print(f'Extracting Google open Roles. Page: {page}')
        urls = get_all_urls(BASE_URL, page)
        current_page_google_job_urls = get_google_job_urls(urls)
        google_job_urls.extend(current_page_google_job_urls)
        page += 1

        if len(current_page_google_job_urls) == 0:
            print('\n\n')
            break

    print(f"Crunching Job Content...")
    contents = get_url_content(google_job_urls)
    google_jobs_content = [
        {
            'link': content.metadata['source'], 
            'title': content.metadata['title'],
            'content': content.page_content,
        } 
        for content in contents
    ]

    return google_jobs_content


def get_model(load_from_hugging_face=False):
    if load_from_hugging_face:
        llm = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-120b",
            task="text-generation",
            provider="auto",  # set your provider here
        )

        return ChatHuggingFace(llm=llm)
    
    return ChatOpenAI(model="gpt-4", temperature=0.0)


def get_resume_content():
    loader = PyPDFLoader('knowledge_base/muskan-resume.pdf')
    pages = []

    for page in loader.lazy_load():
        pages.append(page)
    
    page_content = [page.page_content for page in pages]
    return '\n'.join(page_content)


def get_related_job_roles(open_roles, resume_content):
    model = get_model(load_from_hugging_face=False)
    job_roles = ', '.join([open_role['title'] for open_role in open_roles])

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RELATABLE_JOB_ROLES_PROMPT),
            ("human", "Job Roles: \n\n {job_roles} \n\n Candidate's resume: {resume}"),
        ]
    )

    grade_chain = grade_prompt | model | StrOutputParser()
    response = grade_chain.invoke({"job_roles": job_roles, "resume": resume_content})

    return get_recommended_role_profiles(response.split('\n'), open_roles)


def get_recommended_role_profiles(recommended_role_titles, open_roles):
    print(f"\n\n Recommended Role Titles: {recommended_role_titles}")
    recommended_role_profiles = []
    recommended_role_titles = [
        recommended_role_title.strip().lower() 
        for recommended_role_title in recommended_role_titles
    ]
    
    # print('Recommended Role Titles:', recommended_role_titles)
    for open_role in open_roles:
        # print('Checking Open Role:', open_role['title'])
        if open_role['title'].strip().lower() in recommended_role_titles:
            recommended_role_profiles.append(open_role)

    return recommended_role_profiles


def get_minimum_qualifications(job_profile_content):
    prompt = hub.pull("rlm/rag-prompt")
    llm = get_model(load_from_hugging_face=False)
    rag_chain = prompt | llm | StrOutputParser()
    
    response = rag_chain.invoke(
        {"context": job_profile_content, "question": MINIMUM_QUALIFICATION_QUESTION})

    return response


def get_preferred_qualification(job_profile_content):
    prompt = hub.pull("rlm/rag-prompt")
    llm = get_model(load_from_hugging_face=False)
    rag_chain = prompt | llm | StrOutputParser()
    
    response = rag_chain.invoke(
        {"context": job_profile_content, "question": PREFERRED_QUALIFICATION_QUESTION})

    return response


def get_job_responsibilities(job_profile_content):
    prompt = hub.pull("rlm/rag-prompt")
    llm = get_model(load_from_hugging_face=False)
    rag_chain = prompt | llm | StrOutputParser()
    
    response = rag_chain.invoke(
        {"context": job_profile_content, "question": JOB_RESPONSIBILITIES_QUESTION})

    return response


def analyze_recommended_roles(recommended_roles, resume_content):
    reports = []

    for recommended_role in recommended_roles[:MAX_ROLES]:
        print(f"\n\n Analyzing your Resume for: {recommended_role['title']} ...")

        model = get_model(load_from_hugging_face=False)
        min_qual = get_minimum_qualifications(recommended_role['content'])
        preferred_qual = get_preferred_qualification(recommended_role['content'])
        responsibilities = get_job_responsibilities(recommended_role['content'])

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESUME_ANALYSIS_PROMPT),
                ("human", """
                    Minimum Qualification: {min_qual} \n\n 
                    Preferred Qualification: {preferred_qual} \n\n 
                    Role Responsibilities: {responsibilities} \n\n
                    Candidate's resume: {resume}"""),
            ]
        )

        grade_chain = grade_prompt | model | StrOutputParser()
        response = grade_chain.invoke(
            {
                "min_qual": min_qual, 
                "preferred_qual": preferred_qual,
                "responsibilities": responsibilities,
                "resume": resume_content
            }
        )

        reports.append({
            'title': recommended_role['title'],
            'link': recommended_role['link'],
            'report': response
        })

    return reports


load_dotenv()

google_jobs_content = get_google_jobs_content()
resume_content = get_resume_content()

related_job_roles = get_related_job_roles(google_jobs_content, resume_content)
print(f"\n\n Recommended Job Roles: {[related_job_role['title'] for related_job_role in related_job_roles]}")

if len(related_job_roles) > 0:
    generated_reports = analyze_recommended_roles(related_job_roles, resume_content)
    print('\n\n Report generated!')
    final_report = '# Google SWE Job Role Report (India) \n'
    final_report += f'## This report is generated after analyzing on your Resume for the top {MAX_ROLES} most suited open Google job roles.\n\n'
    final_report += '---\n\n'

    for generated_report in generated_reports:
        final_report += f"## [{generated_report['title']}]({generated_report['link']})\n\n"
        final_report += f"{generated_report['report']}\n\n"
        final_report += '---\n\n'
    
    with open("report/Report.md", "w") as file:
        file.write(final_report)

    print('\n Report saved!')
else:
    print('\n No related job roles found based on the resume content!')
import ast
import re

from langchain import OpenAI
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.agent_toolkits import (
    SQLDatabaseToolkit,
    create_retriever_tool,
    create_sql_agent,
)
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.sql_database import SQLDatabase
from langchain.tools import Tool
from langchain.utilities import SQLDatabase
from langchain.vectorstores import FAISS
from langchain_experimental.sql import SQLDatabaseChain
from pydantic import BaseModel, Field


def run_query_save_results(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r'\b\d+\b', '', string).strip() for string in res]

    return res

def run_query_save_results_names(db, query):
    res = db.run(query)
    res = ast.literal_eval(res)
    res = [' '.join(i) for i in res]

    return res

def get_retriever(texts):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(texts, embeddings)

    return vector_db.as_retriever()

def get_agent():
    
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    llm = ChatOpenAI(temperature=0, model_name='gpt-4')

    artists = run_query_save_results(db, "SELECT Name FROM Artist")
    customers = run_query_save_results(db, "SELECT Company, Address, City, State, Country FROM Customer")
    employees = run_query_save_results(db, "SELECT Address, City, State, Country FROM Employee")
    albums = run_query_save_results(db, "SELECT Title FROM Album")

    customer_names = run_query_save_results_names(db, "SELECT FirstName, LastName FROM Customer")
    employee_names = run_query_save_results_names(db, "SELECT FirstName, LastName FROM Employee")

    texts = (
        artists + 
        customers + 
        customer_names +
        employee_names +
        employees + 
        albums
    )

    retriever = get_retriever(texts)

    retriever_tool = create_retriever_tool(
        retriever,
        name='name_search',
        description='use to learn how a piece of data is actually written, can be 	 from names, surnames addresses etc'
    )

    # sql_agent = create_sql_agent(
    #     llm=llm,
    #     toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    #     verbose=True,
    #     agent_type=AgentType.OPENAI_FUNCTIONS
    # )

    # sql_tool = Tool(
    #     func=sql_agent.run,
    #     name="db_agent",
    #     description="use to get information from the databases, ask exactly what you want in natural language"
    # )

    db_chain = SQLDatabaseChain.from_llm(
        OpenAI(temperature=0, verbose=True),
        db
        )

    sql_tool = Tool(
        func=db_chain.run,
        name="db_agent",
        description="use to get information from the databases, ask exactly what you want in natural language"
    )

    TEMPLATE = """You are working with an SQL database.

        You have a tool called `name_search` through which you can lookup the name of any entity that is present in the database. This could be a person name, an address, a music track name or others.
        You should always use this `name_search` tool to search for the correct way that something is written before you use the `db_agent` tool.
        You should use the `name_search` tool ONLY ONCE and you should also use the `db_agent` tool ONLY ONCE.

        If the user questions contains a term that is not spelled correctly, you should assume that the user meant the correct spelling and answer the question for the correctly spelled term.

        As soon as you have an answer to the question, you should return and not invoke more functions.
    """

    class PythonInputs(BaseModel):
        query: str = Field(description="code snippet to run")

    template = TEMPLATE.format()

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}")
    ])

    tools = [
        sql_tool,
        retriever_tool
    ]

    agent = OpenAIFunctionsAgent(
	    llm=llm,
		prompt=prompt,
		tools=tools
		)
    
    agent_executor = AgentExecutor(
	    agent=agent,
		tools=tools,
		max_iterations=2,
		early_stopping_method="generate"
		)
    return agent_executor
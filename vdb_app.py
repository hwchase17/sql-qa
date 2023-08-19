import ast
import re

import streamlit as st
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
from langsmith import Client
from pydantic import BaseModel, Field

st.set_page_config(page_title='🦜🔗 Ask the VDB/SQL DB App')
st.title('🦜🔗 Ask the VDB/SQL DB App')
st.info("""
Most 'question answering' applications run over unstructured text data. 
But a lot of the data in the world is tabular data! 
This is an attempt to create an application  using [LangChain](https://github.com/langchain-ai/langchain) to let you ask questions of data in tabular format. 
The special property about this application is that it is **robust to spelling mistakes**: you can spell an artist or song wrong but you should still get the results you are looking for.
For this demo application, we will use the Chinook dataset in a SQL database. 
Please explore it [here](https://github.com/lerocha/chinook-database) to get a sense for what questions you can ask. 
Please leave feedback on how well the question is answered, and we will use that improve the application!
""")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(temperature=0, model_name='gpt-4')

@st.cache_data
def run_query_save_results(_db, query):
    res = _db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r'\b\d+\b', '', string).strip() for string in res]

    return res

@st.cache_data
def run_query_save_results_names(_db, query):
    res = _db.run(query)
    res = ast.literal_eval(res)
    res = [' '.join(i) for i in res]

    return res

@st.cache_data
def get_retriever(texts):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(texts, embeddings)

    return vector_db.as_retriever()

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

sql_agent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

sql_tool = Tool(
    func=sql_agent.run,
    name="db_agent",
    description="use to get information from the databases, ask exactly what you want in natural language"
)

TEMPLATE = """You are working with an SQL database.

You have a tool called `name_search` through which you can lookup the name of any entity that is present in the database. This could be a person name, an address, a music track name or others.
You should always use this `name_search` tool to search for the correct way that something is written before you use the `db_agent` tool.

If the user questions contains a term that is not spelled correctly, you should assume that the user meant the correct spelling.

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

def get_chain():
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

chain = get_chain()

chain.verbose = True
chain.return_intermediate_steps = False

from langsmith import Client

client = Client()
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)

query_text = st.text_input('Enter your question:', placeholder = 'How many artists are there?')
# Form input and query
result = None
with st.form('myform', clear_on_submit=True):
	submitted = st.form_submit_button('Submit')
	if submitted:
		with st.spinner('Calculating...'):
			response = chain(query_text, include_run_info=True)
			result = response["output"]
			run_id = response["__run"].run_id
if result is not None:
	st.info(result)
	col_blank, col_text, col1, col2 = st.columns([10, 2,1,1])
	with col_text:
		st.text("Feedback:")
	with col1:
		st.button("👍", on_click=send_feedback, args=(run_id, 1))
	with col2:
		st.button("👎", on_click=send_feedback, args=(run_id, 0))
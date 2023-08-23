import streamlit as st
import langchain
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset
from pydantic import BaseModel, Field

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(temperature=0)
db_chain = SQLDatabaseChain.from_llm(llm, db, return_intermediate_steps=True)

from langsmith import Client
client = Client()
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)

st.set_page_config(page_title='🦜🔗 Ask the SQL DB App')
st.title('🦜🔗 Ask the SQL DB App')
st.info("Most 'question answering' applications run over unstructured text data. But a lot of the data in the world is tabular data! This is an attempt to create an application using [LangChain](https://github.com/langchain-ai/langchain) to let you ask questions of data in tabular format. For this demo application, we will use the Chinook dataset in a SQL database. Please explore the schema [here](https://www.sqlitetutorial.net/wp-content/uploads/2015/11/sqlite-sample-database-color.jpg) to get a sense for what questions you can ask. Please leave feedback on well the question is answered, and we will use that improve the application!")

query_text = st.text_input('Enter your question:', placeholder = 'Ask something like "How many artists are there?" or "Which artist has the most albums"')
# Form input and query
result = None
with st.form('myform', clear_on_submit=True):
	submitted = st.form_submit_button('Submit')
	if submitted:
		with st.spinner('Calculating...'):
			inputs = {"query": query_text}
			response = db_chain(inputs, include_run_info=True)
			result = response["result"]
			sql_command = response["intermediate_steps"][1]
			sql_result = response["intermediate_steps"][3]
			run_id = response["__run"].run_id
if result is not None:
	st.info(result)
	st.code(sql_command)
	st.code(sql_result)
	col_blank, col_text, col1, col2 = st.columns([10, 2,1,1])
	with col_text:
		st.text("Feedback:")
	with col1:
		st.button("👍", on_click=send_feedback, args=(run_id, 1))
	with col2:
		st.button("👎", on_click=send_feedback, args=(run_id, 0))


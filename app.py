import streamlit as st

from utils import get_agent

st.set_page_config(page_title='🦜🔗 Ask the SQL DB App')
st.title('🦜🔗 Ask the SQL DB App')
st.info("""
Most 'question answering' applications run over unstructured text data. 
But a lot of the data in the world is tabular data! 
This is an attempt to create an application  using [LangChain](https://github.com/langchain-ai/langchain) to let you ask questions of data in tabular format. 
The special property about this application is that it is **robust to spelling mistakes**: you can spell an artist or song wrong but you should still get the results you are looking for.
For this demo application, we will use the Chinook dataset in a SQL database. 
Please explore it [here](https://github.com/lerocha/chinook-database) to get a sense for what questions you can ask. 
Please leave feedback on how well the question is answered, and we will use that improve the application!
""")
	
agent = get_agent()

agent.verbose = True
agent.return_intermediate_steps = False

from langsmith import Client

client = Client()
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)

query_text = st.text_input('Enter your question:', placeholder = 'How many artists are there?')

print(query_text)

result = None
with st.form('myform', clear_on_submit=True):
	submitted = st.form_submit_button('Submit')
	if submitted:
		with st.spinner('Calculating...'):
			response = agent(query_text, include_run_info=True)
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
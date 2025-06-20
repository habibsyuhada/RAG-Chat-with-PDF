import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Your AI assistant 🤖", page_icon = "🤖")
st.title("Your AI assistant 🤖")


model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]

def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
	llm = HuggingFaceEndpoint(
		repo_id=model,
		temperature=temperature,
		max_new_tokens=512,
		return_full_text=False,
		task="conversational"  # Explicitly set task
	)
	chat_model = ChatHuggingFace(llm=llm)
	return chat_model

def model_openai(model="gpt-4o-mini", temperature=0.1):
	llm = ChatOpenAi(
		model=model,
		temperature=temperature,
	)
	return llm

def model_ollama(model="phi3", temperature=0.1):
	llm = ChatOllama(
		model=model,
		temperature=temperature,
	)
	return llm

def model_response(user_query, chat_history, model_class):
	if model_class == "hf_hub":
		llm = model_hf_hub()
	elif model_class == "openai":
		llm = model_openai()
	elif model_class == "ollama":
		llm = model_ollama()

	system_prompt = """
	You are a helpful assistant answering general question. Please respond in {language}.
	"""

	language = "the same language the user is using to chat"

	if model_class.startswith("hf"):
		user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
	else:
		user_prompt = "{input}"

	prompt_template = ChatPromptTemplate.from_messages([
		("system", system_prompt),
		MessagesPlaceholder(variable_name="chat_history"),
		("user", user_prompt)
	])

	chain = prompt_template | llm | StrOutputParser()

	return chain.stream({
		"chat_history": chat_history,
		"input": user_query,
		"language": language,
	})

if "chat_history" not in st.session_state:
	st.session_state.chat_history = [
		AIMessage(content="Hi, I'm your virtual assistant! How can I help you?")
	]

for message in st.session_state.chat_history:
	if isinstance(message, AIMessage):
		with st.chat_message("AI"):
			st.write(message.content)
	elif isinstance(message, HumanMessage):
		with st.chat_message("Human"):
			st.write(message.content)

user_query = st.chat_input("Enter your message here...")
if user_query is not None and user_query != "":
	st.session_state.chat_history.append(HumanMessage(content=user_query))

	with st.chat_message("Human"):
		st.markdown(user_query)

	with st.chat_message("AI"):
		resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class))

	st.session_state.chat_history.append(AIMessage(content=resp))
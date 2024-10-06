import os
import io
import json
import datetime
import streamlit as st

import pymupdf

import google.generativeai as genai
from google.generativeai import caching
from google.api_core.exceptions import InvalidArgument


# Show title and description.
st.title("📄 Document question answering from gemini")
st.write(
    "Upload a document below and ask a question about it – gemini will answer! "
    "To use this app, you need to provide an Gemini API key, which you can get [here](https://ai.google.dev/gemini-api/docs/api-key?hl=ja). "
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_messages" not in st.session_state:
    st.session_state.system_messages = ""

st.session_state.system_messages = st.sidebar.text_area(
    "Write your system prompt",
    value = ""
)

gemini_api_key = st.text_input("Gemini API Key", type="password")
if not gemini_api_key:
    st.info("Please add your Gemini API key to continue.", icon="🗝️")
else:

    genai.configure(api_key="AI-xxxxxxxxxxxxxxxx")

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf or .md or .mmd)", type=("pdf", "md", "mmd")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.chat_input(
        "Now ask a question about the document!"
    )

    if question:  # userが入力したときだけメッセージを追加
        st.session_state.messages.append({"role": "user", "content": question})

        # 過去のメッセージを表示する
        for message in st.session_state.messages:
            role = message["role"]
            message_content = message["content"]
            with st.chat_message(role):
                st.markdown(message_content)

    if uploaded_file and question:
        # read file
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type in ['md', 'mmd']:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            content = stringio.read()

        else:
            file_bytes = uploaded_file.read()
            doc = pymupdf.open(stream=io.BytesIO(file_bytes), filetype="pdf")
            all_text = []

            for page in doc:
                text = page.get_text()
                all_text.append(text)

            content = "\n".join(all_text)

        try:
            # Create a cache with a 5 minute TTL
            cache = caching.CachedContent.create(
                model='models/gemini-1.5-pro-001',
                display_name='paper', # used to identify the cache
                system_instruction=(st.session_state.system_messages),
                contents=[content],
                ttl=datetime.timedelta(minutes=10),
            )

            # Construct a GenerativeModel which uses the created cache.
            model = genai.GenerativeModel.from_cached_content(
                cached_content=cache
                )

        except InvalidArgument as e:
            # InvalidArgumentが発生した場合の処理

            model = genai.GenerativeModel(
                model_name='models/gemini-1.5-pro-001',
                system_instruction=(st.session_state.system_messages),
            )

        stringified_messages = [json.dumps(m) for m in st.session_state.messages]
        conversation_history = "\n".join(stringified_messages)

        response_container = st.empty()
        full_response = ""
        if model.cached_content is not None:
            for chunk in model.generate_content(conversation_history, stream=True):
                full_response += chunk.text if chunk is not None else ""
                response_container.markdown(full_response + "▌")
        else:
            for chunk in model.generate_content(content + "\n" + conversation_history, stream=True):
                full_response += chunk.text if chunk is not None else ""
                response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

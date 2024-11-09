import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from pytube.exceptions import PytubeError  # Import PytubeError to handle specific exceptions

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")

# Check if the API key is accessible
if "GROQ_API_KEY" in st.secrets:
    st.write("API Key found in secrets")
else:
    st.write("API Key missing from secrets")

# Streamlit APP Title and Subtitle
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Get the URL (YT or website) to be summarized
generic_url = st.text_input("URL", label_visibility="collapsed")

# Initialize the Gemma model using the Groq API key from Streamlit secrets
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=st.secrets["GROQ_API_KEY"])

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button to start summarization
if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load data from the URL
                if "youtube.com" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        docs = loader.load()
                    except PytubeError as yt_error:
                        st.error("Failed to load YouTube video. Please try a different video or report this issue.")
                        st.stop()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()

                # Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")

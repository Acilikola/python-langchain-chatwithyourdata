'''
RETRIEVAL AUGMENTED GENERATION

In retrieval augmented generation (RAG), an LLM retrieves contextual documents from an external dataset as part of its execution.
This is useful if we want to ask question about specific documents (e.g., our PDFs, a set of videos, etc). 

Overall Workflow: 
1) Document_Loading
2) Splitting
3) Storage (Vectorstore)
4) Retrieval (Query/Question + Relevant_Splits)
5) Output (Prompt -> LLM) > Answer

Loaders deal with the specifics of accessing and converting data. 
'''

#! pip install langchain
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

###PDFs
'''
Let's load a PDF transcript from Andrew Ng's famous CS229 course! 
These documents are the result of automated transcription so words and sentences are sometimes split unexpectedly.
'''

#! pip install pypdf
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
'''
Each page is a Document.
A Document contains text (page_content) and metadata.
'''
len(pages)
page = pages[0]
print(page.page_content[0:500])
page.metadata
###

###YOUTUBE
# ! pip install yt_dlp
# ! pip install pydub
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()

docs[0].page_content[0:500]
###

###URLs
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")
docs = loader.load()
print(docs[0].page_content[:500])
###

###NOTION
'''
Follow steps https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/notion for an example Notion site such as this one: https://yolospace.notion.site/Blendle-s-Employee-Handbook-e31bff7da17346ee99f531087d8b133f
- Duplicate the page into your own Notion space and export as Markdown / CSV.
- Unzip it and save it as a folder that contains the markdown file for the Notion page.
'''
###
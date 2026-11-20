from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


def format_yt_transcript_op(transcript_op):
    snippets = transcript_op.snippets
    result = " ".join( snippet.text for snippet in snippets)
    return result

video_id = "XuoqKYxDHVc"

############################################################################
# Step 1: Fetch the transcript from the give Youtube video ID
############################################################################

yt_api = YouTubeTranscriptApi()
try:
    transcript_op = yt_api.fetch(video_id=video_id, languages=["en"])
except TranscriptsDisabled:
    raise Exception("Transcripts not found for the give video.")

formatted_transcript = format_yt_transcript_op(transcript_op=transcript_op)

############################################################################
# Step 2: Split the formatted_transcript into chunks
############################################################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

transcript_chunks = text_splitter.create_documents([formatted_transcript])

############################################################################
# Step 3: Embedding & create vector store from transcript ocument chunks
############################################################################
load_dotenv()
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(
    documents=transcript_chunks, 
    embedding=embedding
)
vectorstore.save_local("youtube_transcript_index")

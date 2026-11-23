from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from helper import extract_content_from_documents

load_dotenv()

############################################################################
# get vector store access
############################################################################
embedding = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "youtube_transcript_index",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

############################################################################
# take query from the user
############################################################################
query = "Is there anything related to AI discussed in the video? If Yes, please summarize in 5 sentences."

############################################################################
# build MMR (Multip marginal retriever) from vectordb 
############################################################################
retriever = vectorstore.as_retriever(search_kwargs={"k":3,"lambda_mult":0.5},search_type="mmr")
# contextual_result = retriever.invoke(query)
# related_transcript = extract_content_from_documents(contextual_result)

############################################################################
# build prompt template
############################################################################

prompt = PromptTemplate(
    template="""You are helpful Assistant to answer any query.
            Answer for the query from the given transcript context. 
            If you do not find the answer form context, just say I don't know the answer \n
            Transcript: {transcript} \n
            Query: {query}
            """,
    input_variables=["query","transcript"]
)
# formatted_prompt = prompt.invoke({"query":query,"transcript":related_transcript})

############################################################################
# Create LLM model instance for generation
############################################################################

model = ChatOpenAI(model="gpt-5.2")
# model_op = model.invoke(formatted_prompt)

############################################################################
# Join retriever -> prompt -> model -> parser with chain
############################################################################

parallel_chain = RunnableParallel({
    "query": RunnablePassthrough(),
    "transcript": retriever | RunnableLambda(extract_content_from_documents)
})

chain = parallel_chain | prompt | model | StrOutputParser()

result = chain.invoke(query)

print(result)
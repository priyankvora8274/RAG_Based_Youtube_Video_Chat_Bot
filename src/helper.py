from typing import List
from langchain_core.documents import Document

def extract_content_from_documents(documents:List[Document]):
    result = " ".join(document.page_content for document in documents)
    return result
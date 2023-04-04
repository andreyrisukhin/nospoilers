import hashlib, sys
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from timer import Timer

FILENAME = "treasure_island.txt"
COLLECTION_NAME = "engrams"
EMBEDDING_MODEL = "text-embedding-ada-002"
LANGUAGE_MODEL = "gpt-3.5-turbo"

system_message = SystemMessagePromptTemplate.from_template("""
Identify the characters in the following excerpt from the story. For each
character, summarize facts that you learned about the character. If there are
interactions between characters in the excerpt, make sure to make note of
those.
""")
human_message = HumanMessagePromptTemplate.from_template("{text}")
chat_prompt = ChatPromptTemplate.from_messages([system_message, 
                                                human_message])
llm = ChatOpenAI(model_name=LANGUAGE_MODEL, temperature=0)
summary_chain = LLMChain(llm=llm, prompt=chat_prompt)

loader = TextLoader(FILENAME)
content = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=100
)
texts = text_splitter.split_documents(content)

embedder = OpenAIEmbeddings()
ids, embeddings, metadatas = [], [], []
with Timer(f"Computing summaries and embeddings for {len(texts)} texts..."):
    for i in range(len(texts)):
        doc = texts[i]
        extract = texts[i].page_content
        summary = summary_chain.run(extract)
        id = hashlib.sha256(summary.encode("utf-8")).hexdigest()
        metadata = {
            "sequence": i,
            "extract_text": doc.page_content,
            "text": summary
        }
        ids.append(id)
        embeddings.append(embedder.embed_query(summary))
        metadatas.append(metadata)
        print(".", end="")
        sys.stdout.flush()
    print("\n")

with Timer("Inserting embeddings into chromadb..."):
    db = Chroma(COLLECTION_NAME, 
                embedding_function=embedder,
                persist_directory="./.chromadb")
    db.add_texts(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        texts=[text.page_content for text in texts]
    )
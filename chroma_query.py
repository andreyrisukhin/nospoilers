from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.vectorstores import Chroma
import pickle

DEBUG = False
COLLECTION_NAME = "engrams"
EMBEDDING_MODEL = "text-embedding-ada-002"
LANGUAGE_MODEL = "gpt-4"
DOCUMENT_SECTIONS = "sections.pickle"

embedder = OpenAIEmbeddings()

db = Chroma(collection_name=COLLECTION_NAME, 
            embedding_function=embedder,
            persist_directory="./.chromadb")

llm = ChatOpenAI(
    model=LANGUAGE_MODEL,
    temperature=0
)
system_message = SystemMessagePromptTemplate.from_template("""
You are an AI assistant helping a reader to understand the story. You will
answer the question only using relevant facts from the list of facts below.
""")
human_message = HumanMessagePromptTemplate.from_template("{text}")
chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                human_message])
answer_chain = LLMChain(llm=llm, prompt=chat_prompt)

# Each question is a tuple containing the question and the chunk of text
# that defines the point in the book that we're at.
QUESTIONS = [
    ("What does Jim's father do for a living?", ""),
    ("Why does Billy give Jim money each month?", ""),
    ("Where is Billy when he receives the black spot?", ""),
    # "In what century is Treasure Island set?",
    # "What is Pew's most noticeable physical feature?",
    # "To whom does Jim first show the map?"
    # ("What interactions did Jim and Billy Bones have?", 
    #  "But the blind man swore at them again"),
    ("What are the names of the pirates in the story?", 
     "When we were about half-way"),
    ("What are the names of the pirates in the story?",
     ""),
]

with open(DOCUMENT_SECTIONS, "rb") as f:
    docs = pickle.load(f)

for question, text_at_position in QUESTIONS:
    limit = -1
    filter = {}
    if text_at_position != "":
        for sequence, doc in enumerate(docs):
            if text_at_position in doc.page_content:
                limit = sequence
    if limit != -1:
        filter = {
            "sequence": {
                "$lt": limit
            }
        }
    print(f"Limiting to sequence: {limit}") if DEBUG else None
    results = db.similarity_search(question, k=5, filter=filter)
    sequences = set()
    context = ""

    print(f"Received {len(results)} results...") if DEBUG else None
    for result in results:
        sequence = result.metadata["sequence"]
        if not sequence in sequences:
            sequences.add(sequence)
            context += result.metadata['text']
            if DEBUG:
                print(f"Adding sequence {sequence} to context")
                print(f"Facts: {result.metadata['text']}\n\n")

    query = f"QUESTION: {question}\n\nFACTS: {context}"
    answer = answer_chain.run(query)
    print(f"QUESTION: {question}\n\nANSWER: {answer}\n\n")
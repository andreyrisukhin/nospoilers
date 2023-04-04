from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from timer import Timer
import pickle

FILENAME = "treasure_island.txt"
DOCUMENT_SECTIONS = "sections.pickle"

with Timer(f"Processing {FILENAME}..."):
    loader = TextLoader(FILENAME)
    content = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(content)

    for i, text in enumerate(texts):
        text.metadata["sequence"] = i

    with open(DOCUMENT_SECTIONS, 'wb') as f:
        pickle.dump(texts, f, pickle.HIGHEST_PROTOCOL)
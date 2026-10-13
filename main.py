import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 1. Initialize the LLM and Embedding Model
# (Make sure os.environ["OPENAI_API_KEY"] is set)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Prepare raw document text
raw_text = """
LangChain was launched in 2022 as an open-source project by Harrison Chase. 
It provides developers with the tools to connect LLMs to external data sources.
ChromaDB is a popular vector database used alongside LangChain for local development because it is lightweight.
FAISS is another vector index framework created by Meta, built for massive scaling on CPUs and GPUs.
"""

# 3. Split the text into smaller, manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.create_documents([raw_text])

# 4. Embed chunks and store them in ChromaDB (InMemory for this example)
vector_store = Chroma.from_documents(chunks, embeddings)

# 5. Create a retriever out of the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# 6. Define the workflow: Retrieve context -> Ask LLM
def ask_bot(question: str):
    # Retrieve relevant text chunks based on the question
    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the following context:\n\n{context}"),
        ("human", "{question}")
    ])

    # Format prompt and run through the LLM
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


# --- Test the Chain ---
question = "Who created FAISS and what is it used for?"
answer = ask_bot(question)

print(f"Question: {question}")
print(f"Answer: {answer}")
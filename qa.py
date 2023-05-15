# environment setup
import os
# for convenience, I directly used one of my private api key 
os.environ["OPENAI_API_KEY"] = "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"
# import packages
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def qa(file, query):
    # load document
    
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                    persist_directory=".")
    vectordb.persist()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory)
    result = pdf_qa({"question": query})
    # print(result['answer'])
    return result


def main():
    # print("The answer is:", qa(pdf_file_path, query));
    pdf_file_name = input("Enter the name of the PDF file:")
    pdf_file_path = os.path.join(os.getcwd(), pdf_file_name)
    query = input("Enter your query:")
    result = qa(pdf_file_path, query)
    answer = result["answer"]
    border = "*" * (len(answer) + 4)
    print(border)
    print(answer)
    print(border)
    
# Check if the script is being run as the main program
if __name__ == "__main__":
    # Call the main function
    main()

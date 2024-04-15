from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
import argparse
from dataclasses import dataclass

CHROMA_PATH = "chroma"
def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    model = ChatOllama(model="llama2:7b-chat") # Requires Ollama setup and model running
    prompt = PromptTemplate.from_template(
                """
                <s> [INST] You are a chatbot assistant question-answering tasks. 
                You must understand a customers use case and recommend the correct product to their scenario like "Based on your query:".
                Do not say "Based on the context provided" 
                Be direct and confident in your answer and also tackle FAQ or generic questions about the context ONLY.
                Do not answer more than 4 lines.
                Don't justify your answers. 
                Don't give information not mentioned in the CONTEXT INFORMATION. 
                If the answer is not in the context, say the words "Sorry, I am unable to answer your question with the information available to me"
                Use only the following pieces of retrieved context to answer the question. 
                [/INST] </s> 
                [INST] 
                Context: {context}
                Question: {question} 
                Answer: [/INST]
                """
            )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=FastEmbedEmbeddings()) # The threshold risk, if not reached creates a warning, need to find a way to stop processing there instead
    retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 6,
                    "score_threshold": 0.5,
                },
            )
    chain = ({"context": retriever, "question": RunnablePassthrough()}
                          | prompt
                          | model
                          | StrOutputParser())
    #for chunks in chain.stream(query_text): #Uncomment this block and comment the print statement after to get output via stream
                #print(chunks)
    print(chain.invoke(query_text))
    
if __name__ == "__main__":
    main()

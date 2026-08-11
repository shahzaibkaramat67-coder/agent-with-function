from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from function_tool.rage_response import Rag_response
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv

load_dotenv()


def retriver_ChromaDB(query):
    try:
        model = os.getenv(
            "EMBADDING_MODEL_SENTENCE", "sentence-transformers/all-MiniLM-L6-v2"
        )

        reranker_model = os.getenv("RERANKER_MODEL")
        embedding_model = HuggingFaceEmbeddings(model=model)

        persist_directory = os.path.join("data", "chroma_db")
        vectorStore = Chroma(
            embedding_function=embedding_model,
            persist_directory=persist_directory,
        )

        retriever = vectorStore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7},
        )

        template = ChatPromptTemplate.from_template(
            [
                (
                    "system",
                    """ 
                    You are a helpful AI assistant. Use only the provided context to answer the question.
                    If the answer is not present in the context, say 'I could not find the answer in the document.'
                    """,
                ),
                ("human", "user Question: {query}\n\nContext:\n{content}"),
            ]
        )

        relevant_docs = retriever.invoke(query)

        print("relevant_docs", relevant_docs)

        pair = [[query, doc.page_content] for doc in relevant_docs]
        print("relevant_docs  pair", pair)

        reranker = CrossEncoder(reranker_model)

        print("relevant_docs pair reranker  ", reranker)

        rerank_docs_score = reranker.predict(pair)
        print("relevant_docs pair reranker  rerank_docs_score", rerank_docs_score)

        relevant_docs = retriever.invoke(query)
        print(
            "relevant_docs pair reranker  rerank_docs_score  relevant_docs",
            relevant_docs,
        )

        ranked_docs = sorted(
            zip(rerank_docs_score, relevant_docs), key=lambda x: x[0], reverse=True
        )
        print(
            "relevant_docs pair reranker  rerank_docs_score  relevant_docs  ranked_docs",
            ranked_docs,
        )

        most_ralivent_docs = [docs for docs in ranked_docs[:4]]

        print("most_ralivent_docs: ", most_ralivent_docs)

        # print("relevant_docs", relevant_docs)

        content = "\n\n".join(doc.page_content for doc in most_ralivent_docs)

        prompt = template.invoke({"query": query, "content": content})
        return Rag_response(prompt)
    except Exception as exc:
        return f"RAG Error: {str(exc)}"

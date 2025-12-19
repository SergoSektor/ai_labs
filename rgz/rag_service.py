import logging
import textwrap
from typing import List, Tuple, Optional

import chromadb
from chromadb.utils import embedding_functions
import ollama
from ollama._types import ResponseError

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K,
)


class RagService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self._get_or_create_collection()
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self.ollama = ollama.Client(host=OLLAMA_BASE_URL)
        self.model = OLLAMA_MODEL
    
    def _get_or_create_collection(self):
        try:
            return self.client.get_or_create_collection(name=COLLECTION_NAME)
        except Exception as e:
            logging.warning(f"Failed to get collection with default settings: {e}")
            try:
                return self.client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e2:
                logging.warning(f"Failed to get collection, trying to delete and recreate: {e2}")
                try:
                    self.client.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass
                try:
                    return self.client.create_collection(name=COLLECTION_NAME)
                except Exception:
                    return self.client.create_collection(
                        name=COLLECTION_NAME,
                        metadata={"hnsw:space": "cosine"}
                    )

    def retrieve(self, query: str, top_k: int = TOP_K):
        query_embedding = self.embed_fn([query])[0]
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
        except RuntimeError as e:
            if "Cannot open header file" in str(e):
                logging.error("ChromaDB index corrupted. Please delete chroma_db folder and run ingest.py --reset")
                raise RuntimeError(
                    "База данных повреждена. Удалите папку 'chroma_db' и запустите:\n"
                    "python ingest.py --reset"
                ) from e
            raise
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return list(zip(docs, metas, distances))

    def _build_prompt(
        self,
        question: str,
        docs: List[Tuple[str, dict, float]],
    ) -> str:
        context_parts = []
        for doc, meta, _dist in docs:
            src = meta.get("source", "unknown")
            context_parts.append(f"[{src}] {doc}")
        context = "\n\n".join(context_parts)

        system_msg = textwrap.dedent(
            f"""\
            Ты — опытный русскоязычный образовательный ассистент. Отвечай ТОЛЬКО на русском языке.
            
            ПРАВИЛА:
            1. Используй ТОЛЬКО информацию из контекста
            2. Если информации нет — честно скажи об этом
            3. Не выдумывай факты
            4. Будь дружелюбным и полезным
            5. Отвечай кратко и по делу, используй списки и эмодзи для структуры, давай практичные советы, если применимо.
            """
        ).strip()


        user_msg = textwrap.dedent(
            f"""\
            Запрос пользователя: {question}
            
            Контекст из базы знаний:
            {context if context else "Контекст не найден."}
            """
        ).strip()

        return system_msg, user_msg

    def generate_answer(
        self,
        question: str,
    ) -> Tuple[str, List[Tuple[str, dict, float]]]:
        docs = self.retrieve(question)
        system_msg, user_msg = self._build_prompt(
            question, docs
        )

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self.ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                answer = response["message"]["content"]
                return answer, docs
            except ResponseError as exc:
                status = getattr(exc, "status_code", None)
                message = str(exc).strip() or "service unavailable"
                
                if status == 503:
                    if attempt < 2:
                        wait_time = 2 + (attempt * 3)
                        logging.warning(f"Ollama 503, retry {attempt + 1}/3 after {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        last_exc = RuntimeError(
                            "Ollama сервис перегружен или недоступен (503).\n\n"
                            "Попробуйте:\n"
                            "• Подождать 10-30 секунд и повторить запрос\n"
                            "• Проверить, что Ollama запущен: `ollama list`\n"
                            "• Перезапустить Ollama сервис"
                        )
                else:
                    last_exc = RuntimeError(f"Ollama error ({status}): {message}")
                
                raise last_exc
        raise last_exc or RuntimeError("Unknown Ollama error")



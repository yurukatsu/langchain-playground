import os
from typing import AbstractSet, Collection, Iterable, Literal, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from tqdm import tqdm


class FaissGenerator:
    def __init__(
        self, embeddings: Embeddings, text_splitter: RecursiveCharacterTextSplitter
    ):
        """FAISSによるベクターストアを生成するクラス

        Args:
            embeddings (Embeddings): 埋め込みモデル
            text_splitter (RecursiveCharacterTextSplitter): テキスト分割器
        """
        self.embeddings = embeddings
        self.text_splitter = text_splitter

    def _run(
        self,
        text: str,
        metadata: Optional[dict] = None,
        _id: Optional[str] = None,
        **faiss_kwargs,
    ) -> FAISS:
        """実行する

        Args:
            text (str): テキスト
            metadata (Optional[str]): メタデータ
            _id (Optional[str]): ID

        Returns:
            FAISS: FAISS
        """
        # IDの初期化処理
        if _id is not None:
            raise NotImplementedError("Not implemented yet")
        else:
            ids = None
        # テキスト分割と埋め込み
        texts = self.text_splitter.split_text(text)
        text_embeddings = []
        metadatas = []
        for chunk_index, text in tqdm(enumerate(texts)):
            text_embeddings.append(self.embeddings.embed_query(text))
            chunk_metadata = {
                "chunk_index": chunk_index,
            }
            metadatas.append(
                chunk_metadata if metadata is None else metadata.update(chunk_metadata)
            )
        text_embedding_pairs = list(zip(texts, text_embeddings))

        return FAISS.from_embeddings(
            text_embedding_pairs,
            self.embeddings,
            metadatas=metadatas,
            ids=ids,
            **faiss_kwargs,
        )

    def run(
        self,
        text: str,
        metadata: Optional[dict] = None,
        _id: Optional[str] = None,
        **faiss_kwargs,
    ) -> FAISS:
        """実行する

        Args:
            text (str): テキスト
            metadata (Optional[str]): メタデータ
            _id (Optional[str]): ID

        Returns:
            FAISS: FAISS
        """
        return self._run(text, metadata, _id, **faiss_kwargs)


class FaissGeneratorWithTiktokenEncoder(FaissGenerator):
    def __init__(
        self,
        embeddings: Embeddings,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        allowed_special: AbstractSet[str] | Literal["all"] = set(),
        disallowed_special: Collection[str] | Literal["all"] = "all",
        separators: Iterable[str] = ["\n\n", "\n", "   ", " ", ""],
        chunk_size: int = 4096,
        chunk_overlap: int = 512,
        **tiktoken_encoder_kwargs,
    ):
        """
        FAISSによるベクターストアを生成するクラス
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            model_name=model_name,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **tiktoken_encoder_kwargs,
        )
        super().__init__(embeddings, text_splitter)


if __name__ == "__main__":
    import openai
    from dotenv import load_dotenv
    from langchain.embeddings import OpenAIEmbeddings

    # Load .env
    dotenv_path = "../../.env"
    load_dotenv(dotenv_path=dotenv_path, override=True)
    # openai settings
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # embeddings
    embeddings = OpenAIEmbeddings()

    # test
    text = (
        "Albert Einstein was born on March 14, 1879, in Ulm, Württemberg, Germany."
        "He grew up in a secular Jewish family."
        "His father, Hermann Einstein, was a salesman and engineer who, with his brother, founded Elektrotechnische Fabrik J. Einstein & Cie, a Munich-based company that mass-produced electrical equipment."
        "Einstein's mother, the former Pauline Koch, ran the family household."
        "Einstein had one sister, Maja, born two years after him."
        "Einstein attended elementary school at the Luitpold Gymnasium in Munich. However, he felt alienated there and struggled with the institution's rigid pedagogical style."
        "He also had what were considered speech challenges."
        "However, he developed a passion for classical music and playing the violin, which would stay with him into his later years."
        "Most significantly, Einstein's youth was marked by deep inquisitiveness and inquiry."
    )
    faiss_generator = FaissGeneratorWithTiktokenEncoder(
        embeddings,
        chunk_size=64,
        chunk_overlap=16,
    )
    faiss = faiss_generator.run(text)
    query = "What is the name of Einstein's father?"
    answer = faiss.similarity_search_with_score(query)
    print(answer)

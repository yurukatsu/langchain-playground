from typing import AbstractSet, Collection, Iterable, Literal, Optional

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from modules.embeddings.text2faiss import FaissGenerator


class FaissGeneratorWithPyPDFLoader(FaissGenerator):
    def __init__(
        self, embeddings: Embeddings, text_splitter: RecursiveCharacterTextSplitter
    ):
        """FAISSによるベクターストアを生成するクラス

        Args:
            embeddings (Embeddings): 埋め込みモデル
            text_splitter (RecursiveCharacterTextSplitter): テキスト分割器
        """
        super().__init__(embeddings, text_splitter)

    def run(
        self,
        file_path: str,
        separator: str = "\n",
        metadata: Optional[dict] = None,
        _id: Optional[str] = None,
        pypdf_loader_kwargs: dict = {},
        faiss_kwargs: dict = {},
    ) -> FAISS:
        loader = PyPDFLoader(file_path, **pypdf_loader_kwargs)
        pages = loader.load_pages()
        text = separator.join([page.page_content for page in pages])
        return self._run(text, metadata=metadata, _id=_id, **faiss_kwargs)


class FaissGeneratorWithPyPDFLoaderWithTikToken(FaissGeneratorWithPyPDFLoader):
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

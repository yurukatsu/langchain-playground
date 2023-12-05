from typing import List

from langchain.vectorstores.faiss import FAISS

def marge_faiss(faiss_list: List[FAISS]) -> FAISS:
    """FAISSをマージする

    Args:
        faiss_list (List[FAISS]): FAISSのリスト

    Returns:
        FAISS: マージされたFAISS
    """
    for i, faiss in enumerate(faiss_list[1:]):
        if  i == 0:
            faiss_all = faiss_list[0]
        else:
            faiss_all.merge_from(faiss)
    return faiss_all

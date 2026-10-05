"""Chroma 内存型数据库测试脚本。

使用内存型（Ephemeral）客户端演示 Chroma 的基本流程：
插入文档 -> 语义检索 -> 直接打印返回结果。

说明：
- 内存型客户端 `chromadb.Client()` 不会将数据落盘，进程结束后即清空，
  适合快速演示与冒烟验证。
- 为避免联网下载 Chroma 默认 embedding 模型（约 83MB，国内网络较慢），
  这里复用本地已缓存的 sentence-transformers 模型
  （paraphrase-multilingual-MiniLM-L12-v2，384 维）作为 embedding function；
  Chroma 仍会自动对 documents / query_texts 进行向量化，无需手动传入 embeddings。
"""

import glob
import json
import os

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# 待插入的文档与对应 id
DOCUMENTS = [
    "This is a document about pineapple",
    "This is a document about oranges",
    "This is a document about cat"
    
]
IDS = ["doc_1", "doc_2", "doc_3"]

# 查询文本：Chroma 会自动对其进行 embedding
QUERY_TEXTS = ["This is a query document about hawaii"]
N_RESULTS = 3

# 本地已缓存的 sentence-transformers 模型快照目录（避免联网下载）
_MODEL_SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/*"
)


class LocalSentenceTransformerEF(EmbeddingFunction[Documents]):
    """基于本地已缓存模型的 embedding 函数，全程不联网。"""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        snapshots = sorted(glob.glob(_MODEL_SNAPSHOT_GLOB))
        if not snapshots:
            raise FileNotFoundError(
                "未找到本地缓存的 sentence-transformers 模型，请先运行 "
                "sentence_similarity.py 下载模型。"
            )
        self._model = SentenceTransformer(snapshots[0])

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(input).tolist()


def _print(title, obj):
    """以标题 + JSON 的形式打印任意可序列化对象。"""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    """运行 Chroma 内存型数据库的插入与查询流程并打印结果。"""
    # 1. 创建内存型客户端（不落盘，禁用匿名遥测）
    client = chromadb.Client(
        chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
    )

    # 2. 创建集合，指定本地 embedding 函数；距离度量用 cosine（文本语义检索最常用）
    collection = client.create_collection(
        name="test_collection",
        embedding_function=LocalSentenceTransformerEF(),
        metadata={"hnsw:space": "cosine"},
    )
    print("已创建集合: test_collection")
    print("插入前文档数量:", collection.count())

    # 3. 插入文档（Chroma 会自动为其生成 embedding）
    collection.add(documents=DOCUMENTS, ids=IDS)
    print("插入后文档数量:", collection.count())

    # 4. 查询（Chroma 会自动对 query_texts 生成 embedding 并做语义检索）
    results = collection.query(query_texts=QUERY_TEXTS, n_results=N_RESULTS)

    # 5. 打印原始返回结果
    _print("query 原始返回结果", results)

    # 6. 以易读的形式逐条打印最相似结果
    print(f"\n{'-' * 60}")
    print(f"查询文本: {QUERY_TEXTS[0]}")
    print(f"最相似 Top-{N_RESULTS} 结果:")
    print(f"{'-' * 60}")
    for rank, (doc_id, doc, dist) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["distances"][0]),
        start=1,
    ):
        print(f"  Top {rank}: id={doc_id}, distance={dist:.6f}")
        print(f"        document={doc}")


if __name__ == "__main__":
    main()

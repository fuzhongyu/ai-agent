"""计算三组句子的 Sentence-Transformers embedding 余弦相似度。"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_DIR / "similarity_scores.json"
ENV_FILE = PROJECT_DIR / ".env"

SENTENCE_PAIRS = [
    ("今天天气很好，适合去公园散步。", "阳光明媚的日子，很适合到公园走走。"),
    ("我想订一张明天去上海的机票。", "请帮我查询明日前往上海的航班。"),
    ("这家餐厅的意大利面非常好吃。", "量子计算机利用量子比特进行运算。"),
]


def load_project_env() -> None:
    """加载项目 .env 中尚未设置的变量，不依赖额外的 dotenv 包。"""
    if not ENV_FILE.is_file():
        return

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        # 兼容 `export KEY=value` 写法，否则会把 "export KEY" 整体当作变量名。
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def load_model() -> "SentenceTransformer":
    """加载模型，并在首次下载失败时给出可操作的错误信息。"""
    # huggingface_hub 在导入时读取 HF_ENDPOINT，因此必须先加载 .env。
    load_project_env()
    from sentence_transformers import SentenceTransformer

    model_source = os.environ.get("SENTENCE_TRANSFORMER_MODEL_PATH", MODEL_NAME)
    try:
        return SentenceTransformer(model_source)
    except OSError as error:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        raise RuntimeError(
            "无法加载句向量模型。请检查网络，或在 .env 中设置 "
            "SENTENCE_TRANSFORMER_MODEL_PATH 为已下载模型目录；"
            f"当前模型来源为 {model_source!r}，HF_ENDPOINT 为 {endpoint!r}。"
        ) from error


def main() -> None:
    model = load_model()
    results = []

    for index, (sentence_a, sentence_b) in enumerate(SENTENCE_PAIRS, start=1):
        embeddings = model.encode([sentence_a, sentence_b], normalize_embeddings=True)
        # 已归一化的向量，点积即余弦相似度，范围约为 [-1, 1]。
        score = float(embeddings[0] @ embeddings[1])
        results.append(
            {
                "group": index,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "cosine_similarity": round(score, 6),
            }
        )

    OUTPUT_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for item in results:
        print(f"第 {item['group']} 组相似度: {item['cosine_similarity']:.6f}")
    print(f"结果已保存至: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

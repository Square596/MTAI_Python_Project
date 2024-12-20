import os
from time import ctime, sleep

import numpy as np
import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sklearn.datasets import load_digits

from app.const import INDEX_NAME, VECTOR_DIM

MNIST_SCHEMA = (
    TagField("$.label", as_name="label"),
    VectorField(
        "$.image",
        "HNSW",
        {
            "TYPE": "FLOAT16",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="image",
    ),
)


def load_mnist():
    print(ctime(), "Start downloading MNIST dataset")

    # mnist = fetch_openml("mnist_784", version=1)
    mnist = load_digits()

    print(ctime(), "MNIST dataset has been downloaded succesfully")

    images = mnist.data.astype(np.float16)
    labels = mnist.target.astype(np.str_)

    return images, labels


def load_data_to_redis(r, images, labels):
    print(ctime(), f"Start uploading {len(images)} images to Redis")

    start_db_size = r.dbsize()

    pipeline = r.pipeline()
    for i, (image, label) in enumerate(zip(images, labels), start=1):
        redis_key = f"{INDEX_NAME}:{start_db_size + i :05}"

        store_label = {
            "label": label,
        }

        pipeline.json().set(redis_key, "$", store_label)
        pipeline.json().set(redis_key, "$.image", image)

        print(ctime(), f"Requested to upload an image. {redis_key=}")

    print(ctime(), "Executing requests")
    pipeline.execute()

    assert r.dbsize() - start_db_size == len(labels), "there is missing data"
    print(ctime(), f"All images have been added to Redis {INDEX_NAME}")
    print(ctime(), f"Current DB size: {r.dbsize()}")


def create_index(r):
    print(ctime(), "Start creating the index")

    definition = IndexDefinition(prefix=[f"{INDEX_NAME}:"], index_type=IndexType.JSON)
    status = r.ft("idx:mnist_vss").create_index(
        fields=MNIST_SCHEMA, definition=definition
    )

    info = r.ft("idx:mnist_vss").info()
    num_docs = info["num_docs"]
    indexing_failures = info["hash_indexing_failures"]

    print(ctime(), "Waiting for the index construction...")
    while num_docs != r.dbsize():
        sleep(0.05)
        info = r.ft("idx:mnist_vss").info()
        num_docs = info["num_docs"]
        indexing_failures = info["hash_indexing_failures"]

        print(ctime(), f"{num_docs / r.dbsize() * 100 :.1f}%...")

    print(ctime(), f"Index has been created. {status=}")

    print(ctime(), f"{num_docs} documents indexed with {indexing_failures} failures")


def predict_label(r, image):
    rquery = (
        Query("(*)=>[KNN 1 @image $query_image AS score]")
        .sort_by("score")
        .return_fields("score", "label")
        .dialect(2)
    )

    result_doc = (
        r.ft("idx:mnist_vss").search(rquery, {"query_image": image.tobytes()}).docs[0]
    )
    return {"score": round(1 - float(result_doc.score), 2), "label": result_doc.label}


if __name__ == "__main__":
    redis_host = os.getenv("REDIS_HOST", "localhost")
    r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
    r.flushdb()  # clean the redis db

    images, labels = load_mnist()
    load_data_to_redis(r, images, labels)
    create_index(r)

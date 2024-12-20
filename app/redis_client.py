import redis
import numpy as np
from sklearn.datasets import load_digits
from app.const import REDIS_HOST, REDIS_PORT, INDEX_NAME, VECTOR_DIM
from redis.commands.search.field import (
    TagField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from time import ctime
import json


MNIST_SCHEMA = (
    TagField("$.label", as_name="label"),
    VectorField(
        "$.embedding",
        "HNSW",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="embedding",
    ),
)


def load_data_to_redis(r, images, labels):
    print(ctime(), f"Start uploading {len(images)} images to Redis")

    current_db_size = r.dbsize()

    pipeline = r.pipeline()
    for i, (image, label) in enumerate(zip(images, labels), start=1):
        redis_key = f"{INDEX_NAME}:{current_db_size + i :05}"

        store_label = {
            "label": label,
        }

        pipeline.json().set(redis_key, "$", store_label)
        pipeline.json().set(redis_key, "$.image", image)

        print(ctime(), f"Requested to upload an image. {redis_key=}")
    
    print(ctime(), "Executing requests")
    pipeline.execute()

    assert r.dbsize() == len(labels), "there is missing data"
    print(ctime(), f"All images have been added to Redis {INDEX_NAME}")

    # print(ctime(), "Start uploading images")

    # pipeline = r.pipeline()
    # for i, image in enumerate(images, start=1):
    #     redis_key = f"{INDEX_NAME}:{current_db_size + i :05}"

    #     pipeline.json().set(redis_key, "$.image", image.tolist())

    # pipeline.execute()

    # print(ctime(), f"All images have been added to Redis {INDEX_NAME}")


    # for i, (image, label) in enumerate(zip(images, labels)):
    #     key = f"{INDEX_NAME}:{current_db_size + i}"
    #     r.hset(
    #         key,
    #         mapping={
    #             "label": label,
    #             "embedding": np.array(image, dtype=np.float32).tobytes(),
    #         },
    #     )
    #     print(ctime(), f"{current_db_size + i} has been added to Redis {INDEX_NAME}")
    print(ctime(), f"Current DB size: {r.dbsize()}")


def create_index(r):
    print(ctime(), "Start creating the index")

    try:
        definition = IndexDefinition(prefix=[f"{INDEX_NAME}:"], index_type=IndexType.JSON)
        status = r.ft("idx:mnist_vss").create_index(fields=MNIST_SCHEMA, definition=definition)
        # r.ft(INDEX_NAME).create_index(MNIST_SCHEMA)

        print(ctime(), f"Index has been created. {status=}")
    except Exception as e:
        print(ctime(), f"Index creation error: {e}")


def load_mnist():
    print(ctime(), "Start downloading MNIST dataset")

    # mnist = fetch_openml("mnist_784", version=1)
    mnist = load_digits()

    print(ctime(), "MNIST dataset has been downloaded succesfully")

    print(ctime(), "Converting to FP32")
    images = mnist.data.astype(np.float32).tolist()
    labels = mnist.target.astype(np.str_).tolist()
    print(ctime(), "Converting completed")

    return images, labels


# if __name__ == "__main__":

#     r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
#     r.flushdb()  # clean the redis db

#     images, labels = load_mnist(logger)
#     load_data_to_redis(r, images, labels, logger)
#     create_index(r)
import redis
import os

from app.redis_client import create_index, load_mnist, load_data_to_redis


redis_host = os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)

# create_index(redis_client)
images, labels = load_mnist()
load_data_to_redis(redis_client, images, labels)
print(f"Current DB size: {redis_client.dbsize()}")

del images
del labels

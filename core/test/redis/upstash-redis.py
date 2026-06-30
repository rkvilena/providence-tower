import redis


def test_upstash_redis_connection():
    # Replace with your actual Upstash Redis URL and token
    url = ""  # fill with upstash token
    r = redis.Redis.from_url(url)
    r.set("foo", "bar")
    return r.get("foo")


print(
    "Result: ", test_upstash_redis_connection()
)  # Should print b'bar' if the connection is successful

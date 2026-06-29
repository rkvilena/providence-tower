import redis


def test_upstash_redis_connection():
    # Replace with your actual Upstash Redis URL and token
    url = "rediss://default:AYy1AAIgcDE4NGE0NzZhNzYzOTA0NDNhYWVkY2ZhNTZjYmI1Zjg1ZA@measured-ibex-36021.upstash.io:6379"
    r = redis.Redis.from_url(url)
    r.set("foo", "bar")
    return r.get("foo")


print(
    "Result: ", test_upstash_redis_connection()
)  # Should print b'bar' if the connection is successful

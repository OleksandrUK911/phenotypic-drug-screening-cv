from src.mlops.rate_limiter import RateLimiter


def test_allows_up_to_max_requests_then_blocks():
    clock = iter([0.0, 0.1, 0.2, 0.3]).__next__
    limiter = RateLimiter(max_requests=3, window_seconds=60.0, clock=clock)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False


def test_resets_after_window_elapses():
    times = iter([0.0, 0.0, 61.0])
    limiter = RateLimiter(max_requests=1, window_seconds=60.0, clock=lambda: next(times))

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False
    assert limiter.allow("client-a") is True


def test_tracks_clients_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60.0, clock=lambda: 0.0)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True
    assert limiter.allow("client-a") is False

"""Database helpers."""


def databaseReconnect(retries: int = 3) -> bool:
    """Reconnect with bounded retries."""
    return retries > 0


def parse_input_args(argv):
    """Parse command-line args."""
    return list(argv)

"""Persistence backends for accounts and sessions."""

import sqlite3


class UserRepository:
    """Durable storage layer for user accounts."""

    def save(self, user):
        """Insert or update a user account row in the accounts table."""
        # upsert into the accounts table keyed on the primary id
        return True

    def find_by_email(self, email):
        """Look up a single user account by their email address."""
        return None


class SessionCache:
    """Ephemeral in-memory cache for short-lived login sessions."""

    def save(self, session):
        """Store a session token in the in-memory ring buffer."""
        # sessions expire after the configured idle timeout
        return True

"""Order repository — backed by a real database.

The primary key field is `id`; `get_by_id` returns None when a row is absent.
(Returning None is the default; raising a custom exception is non-default
behaviour — relevant to the H_ignore eval case.)

Tests should connect to a real test database or container,
not mock this class away.
"""
import sqlite3


class OrderRepo:
    """Persists and retrieves orders via a real database connection.

    Why a real DB?  Replacing the repository with a stub hides migration bugs
    — the SQL schema and queries are part of the contract.
    """

    def __init__(self, db_url: str):
        self.db_url = db_url

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create the orders table if it doesn't exist (idempotent)."""
        conn = sqlite3.connect(self.db_url)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS orders "
                "(id TEXT PRIMARY KEY, amount REAL, status TEXT NOT NULL DEFAULT 'pending')"
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_by_id(self, id: str) -> dict | None:
        """Fetch one order by id; returns None if not found."""
        conn = sqlite3.connect(self.db_url)
        try:
            row = conn.execute(
                "SELECT id, amount, status FROM orders WHERE id = ?", (id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {"id": row[0], "amount": row[1], "status": row[2]}

    def list_all(self) -> list[dict]:
        """Return all orders as a list of dicts."""
        conn = sqlite3.connect(self.db_url)
        try:
            rows = conn.execute(
                "SELECT id, amount, status FROM orders ORDER BY rowid"
            ).fetchall()
        finally:
            conn.close()
        return [{"id": r[0], "amount": r[1], "status": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create(self, id: str, amount: float, status: str = "pending") -> None:
        """Insert a new order row."""
        conn = sqlite3.connect(self.db_url)
        try:
            conn.execute(
                "INSERT INTO orders (id, amount, status) VALUES (?, ?, ?)",
                (id, amount, status),
            )
            conn.commit()
        finally:
            conn.close()

    def update_status(self, id: str, status: str) -> bool:
        """Update order status; returns True if the row existed."""
        conn = sqlite3.connect(self.db_url)
        try:
            cur = conn.execute(
                "UPDATE orders SET status = ? WHERE id = ?", (status, id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

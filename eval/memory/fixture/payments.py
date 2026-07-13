"""Payment service with an external gateway dependency.

Tests for PaymentService should mock / stub the gateway — external calls are
unreliable in CI and incur real charges.  This class has nothing to do with
the database-backed OrderRepo.
"""


class PaymentGateway:
    """External payment gateway (makes real network calls)."""

    def charge(self, amount: float, token: str) -> dict:
        raise NotImplementedError("real network call — stub this in tests")


class PaymentService:
    """Processes a payment via the external gateway.

    Gateway calls are unreliable and slow; stub the gateway in tests.
    """

    def __init__(self, gateway: PaymentGateway):
        self.gateway = gateway

    def process(self, amount: float, token: str) -> bool:
        """Attempt to charge `amount`; returns True on success."""
        if amount <= 0:
            raise ValueError(f"amount must be positive, got {amount}")
        result = self.gateway.charge(amount, token)
        return result.get("status") == "ok"

    def refund(self, amount: float, token: str) -> bool:
        """Refund a previously charged amount."""
        result = self.gateway.charge(-amount, token)
        return result.get("status") == "ok"

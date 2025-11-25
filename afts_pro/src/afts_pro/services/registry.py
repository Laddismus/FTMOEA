class ServiceRegistry:
    """
    Registry placeholder for dependency wiring.
    """

    def __init__(self) -> None:
        self._services = {}

    def register(self, key: str, service: object) -> None:
        self._services[key] = service

    def resolve(self, key: str) -> object:
        return self._services.get(key)

import abc


class BaseRepository(abc.ABC):
    """
    Base class for data repositories (database, file, API).
    """

    @abc.abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def disconnect(self) -> None:
        raise NotImplementedError

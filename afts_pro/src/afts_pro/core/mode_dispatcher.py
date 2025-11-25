from enum import Enum
import logging
from typing import Awaitable, Callable, Dict


class Mode(str, Enum):
    TRAIN = "train"
    SIM = "sim"
    LIVE = "live"


ModeRunner = Callable[[Mode], Awaitable[None]]


class ModeDispatcher:
    """
    Routes CLI-selected mode into the appropriate startup path.
    """

    def __init__(self, runner: ModeRunner) -> None:
        self._runner = runner
        self._handlers: Dict[Mode, Callable[[], Awaitable[None]]] = {
            Mode.TRAIN: self._dispatch_train,
            Mode.SIM: self._dispatch_sim,
            Mode.LIVE: self._dispatch_live,
        }

    async def dispatch(self, mode: Mode) -> None:
        handler = self._handlers.get(mode)
        if handler is None:
            raise ValueError(f"Unsupported mode: {mode}")

        logging.info("Dispatching startup for mode=%s", mode.value)
        await handler()

    async def _dispatch_train(self) -> None:
        await self._runner(Mode.TRAIN)

    async def _dispatch_sim(self) -> None:
        await self._runner(Mode.SIM)

    async def _dispatch_live(self) -> None:
        await self._runner(Mode.LIVE)

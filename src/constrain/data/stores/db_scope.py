# verity-core/verity_core/db/stores/db_scope.py
from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def session_scope(session_maker):
    s = session_maker()
    try:
        yield s
        s.commit()
    except Exception as e:
        log.error(f"Session error: {e}")
        s.rollback()
        raise
    finally:
        s.close()


def retry(fn: Callable[[], T], *, tries: int = 2, base_delay_s: float = 0.15) -> T:
    """
    Tiny retry helper to handle transient DB errors and occasional locking.
    Keeps behavior deterministic and short.
    """
    last_err: Exception | None = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt == tries - 1:
                raise
            # jitter backoff
            time.sleep(base_delay_s * (2**attempt) + random.random() * 0.05)
    # unreachable, but for type-checkers
    raise last_err  # type: ignore[misc]

"""connect util for webapi."""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Optional

from requests import ReadTimeout
from requests.exceptions import ConnectionError as ConnErr
from requests.exceptions import JSONDecodeError
from urllib3.exceptions import NewConnectionError

from tidy3d.exceptions import WebError
from tidy3d.log import log
from tidy3d.web import common
from tidy3d.web.common import REFRESH_TIME


def wait_for_connection(
    decorated_fn: Optional[Callable[..., Any]] = None,
    wait_time_sec: Optional[float] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
    """Causes function to ignore connection errors and retry for ``wait_time_sec`` secs."""

    def decorator(
        web_fn: Callable[..., Any],
        wait_time_sec: Optional[float] = wait_time_sec,
    ) -> Callable[..., Any]:
        """Decorator returned by @wait_for_connection()"""

        @wraps(web_fn)
        def web_fn_wrapped(*args: Any, **kwargs: Any) -> Any:
            """Function to return including connection waiting."""
            time_start = time.time()
            warned_previously = False

            timeout = common.CONNECTION_RETRY_TIME if wait_time_sec is None else wait_time_sec

            while (time.time() - time_start) < timeout:
                try:
                    return web_fn(*args, **kwargs)
                except (ConnErr, ConnectionError, NewConnectionError, ReadTimeout, JSONDecodeError):
                    if not warned_previously:
                        log.warning(f"No connection: Retrying for {timeout} seconds.")
                        warned_previously = True
                    time.sleep(REFRESH_TIME)

            raise WebError("No internet connection: giving up on connection waiting.")

        return web_fn_wrapped

    if decorated_fn:
        return decorator(decorated_fn, wait_time_sec=wait_time_sec)

    return decorator


def get_time_steps_str(time_steps: int) -> str:
    """get_time_steps_str"""
    if time_steps < 1000:
        time_steps_str = f"{time_steps}"
    elif 1000 <= time_steps < 1000 * 1000:
        time_steps_str = f"{time_steps / 1000}K"
    else:
        time_steps_str = f"{time_steps / 1000 / 1000}M"
    return time_steps_str


def get_grid_points_str(grid_points: int) -> str:
    """get_grid_points_str"""
    if grid_points < 1000:
        grid_points_str = f"{grid_points}"
    elif 1000 <= grid_points < 1000 * 1000:
        grid_points_str = f"{grid_points / 1000}K"
    else:
        grid_points_str = f"{grid_points / 1000 / 1000}M"
    return grid_points_str

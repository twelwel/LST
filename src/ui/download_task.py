"""下载任务与 Tk 主线程之间的事件桥。

Tkinter 不是线程安全的。下载线程直接调用 ``widget.after()`` /
``winfo_exists()`` 时，在网络回调密集、快速重复点击或关闭窗口的场景下
可能触发 ``main thread is not in main loop`` 或 Tcl 崩溃。

本模块约定：
- 后台线程只执行 ``work``，并把进度/状态/结束事件写入 Queue；
- 所有控件更新都在主线程的 ``after`` 轮询里完成；
- ``close()`` 之后不再触发任何 UI 回调。
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional

from logger import get_logger


log = get_logger('ui.download_task')


ProgressCallback = Callable[..., None]
StatusCallback = Callable[[str], None]
WorkFunction = Callable[[ProgressCallback, StatusCallback], Any]
DoneCallback = Callable[[bool, Any, Optional[str], bool], None]


def _worker_main(event_queue: queue.Queue,
                 closed_event: threading.Event,
                 cancel_event: threading.Event,
                 work: WorkFunction) -> None:
    """后台线程入口。

    只捕获 Queue / Event / work，不持有 DownloadTask，也不持有 owner。
    这样即使窗口已销毁，线程结束时也不会在后台线程释放 Tk 对象。
    """
    def progress_cb(*args):
        if closed_event.is_set() or cancel_event.is_set():
            return
        try:
            event_queue.put_nowait(('progress', args))
        except Exception:
            pass

    def status_cb(message: str):
        if closed_event.is_set() or cancel_event.is_set():
            return
        try:
            event_queue.put_nowait(('status', str(message)))
        except Exception:
            pass

    if closed_event.is_set():
        return
    if cancel_event.is_set():
        try:
            event_queue.put_nowait(('done', False, None, None, True))
        except Exception:
            pass
        return

    result = None
    error: Optional[str] = None
    try:
        result = work(progress_cb, status_cb)
    except Exception as exc:
        error = str(exc).strip() or exc.__class__.__name__
        log.exception(f'[下载任务] 下载线程异常：{error}')

    if closed_event.is_set():
        return

    cancelled = cancel_event.is_set()
    try:
        event_queue.put_nowait((
            'done',
            bool(result) and error is None and not cancelled,
            result,
            error,
            cancelled,
        ))
    except Exception:
        pass


class DownloadTask:
    """在主线程中安全消费下载线程事件。

    ``on_done(ok, result, error, cancelled)`` 始终在 Tk 主线程执行。
    ``work(progress_cb, status_cb)`` 在后台线程执行，必须只做网络/IO，
    不能访问任何 Tk 控件。

    一个实例只用于一次下载；结束后会自动关闭，下一次下载重新创建即可。
    """

    def __init__(self, owner, *, on_progress: Optional[ProgressCallback] = None,
                 on_status: Optional[StatusCallback] = None,
                 on_done: Optional[DoneCallback] = None,
                 interval_ms: int = 100):
        self._owner = owner
        self._on_progress = on_progress
        self._on_status = on_status
        self._on_done = on_done
        self._interval_ms = max(30, int(interval_ms))

        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._after_id = None
        self._closed = False
        self._cancel_requested = False
        self._closed_event = threading.Event()
        self._cancel_event = threading.Event()
        self._cancel_cb: Optional[Callable[[], None]] = None
        self._state_lock = threading.Lock()

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    @property
    def cancelled(self) -> bool:
        with self._state_lock:
            return self._cancel_requested

    def start(self, work: WorkFunction,
              cancel_cb: Optional[Callable[[], None]] = None) -> bool:
        """启动任务；只能从 Tk 主线程调用。"""
        with self._state_lock:
            if self._closed:
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._cancel_requested = False
            self._cancel_event.clear()
            self._closed_event.clear()
            self._cancel_cb = cancel_cb
            self._thread = threading.Thread(
                target=_worker_main,
                args=(self._queue, self._closed_event,
                      self._cancel_event, work),
                daemon=True,
                name='ModelDownload',
            )

        try:
            self._after_id = self._owner.after(
                self._interval_ms, self._poll)
        except Exception as exc:
            log.exception(f'[下载任务] 无法启动进度轮询：{exc}')
            with self._state_lock:
                self._closed = True
                self._closed_event.set()
                self._thread = None
            return False

        try:
            self._thread.start()
        except Exception as exc:
            log.exception(f'[下载任务] 无法启动下载线程：{exc}')
            self.close()
            return False
        return True

    def cancel(self) -> None:
        """请求取消；不会等待阻塞中的网络请求。"""
        with self._state_lock:
            self._cancel_requested = True
            self._cancel_event.set()
            cancel_cb = self._cancel_cb

        if cancel_cb is not None:
            try:
                cancel_cb()
            except Exception as exc:
                log.warning(f'[下载任务] 取消失败：{exc}')

    def close(self) -> None:
        """停止事件轮询并请求取消；可从 ``destroy`` 中调用。"""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._cancel_requested = True
            self._closed_event.set()
            self._cancel_event.set()
            cancel_cb = self._cancel_cb

        after_id = self._after_id
        self._after_id = None
        if after_id is not None:
            try:
                self._owner.after_cancel(after_id)
            except Exception:
                pass

        if cancel_cb is not None:
            try:
                cancel_cb()
            except Exception:
                pass

        # 不 join：snapshot_download() 可能阻塞在网络请求里。
        # 线程是 daemon，且 close 后不会再访问任何 Tk 对象。

    def _poll(self) -> None:
        self._after_id = None
        if self._closed:
            return
        try:
            if not self._owner.winfo_exists():
                self._closed = True
                self._closed_event.set()
                return
        except Exception:
            self._closed = True
            self._closed_event.set()
            return

        latest_progress = None
        statuses = []
        final = None

        try:
            while True:
                event = self._queue.get_nowait()
                kind = event[0]
                if kind == 'progress':
                    latest_progress = event[1]
                elif kind == 'status':
                    statuses.append(event[1])
                elif kind == 'done':
                    final = event[1:]
        except queue.Empty:
            pass
        except Exception as exc:
            log.exception(f'[下载任务] 读取事件失败：{exc}')

        if latest_progress is not None and self._on_progress:
            try:
                self._on_progress(*latest_progress)
            except Exception as exc:
                log.exception(f'[下载任务] 进度回调失败：{exc}')

        if self._on_status:
            for text in statuses:
                if self._closed:
                    break
                try:
                    self._on_status(text)
                except Exception as exc:
                    log.exception(f'[下载任务] 状态回调失败：{exc}')

        if self._closed:
            return

        if final is not None:
            with self._state_lock:
                self._closed = True
                self._closed_event.set()
                self._cancel_cb = None
            try:
                if self._on_done:
                    self._on_done(*final)
            except Exception as exc:
                log.exception(f'[下载任务] 完成回调失败：{exc}')
            self._thread = None
            return

        self._schedule()

    def _schedule(self) -> None:
        if self._closed:
            return
        try:
            self._after_id = self._owner.after(
                self._interval_ms, self._poll)
        except Exception as exc:
            log.exception(f'[下载任务] 调度进度轮询失败：{exc}')
            self._closed = True
            self._closed_event.set()

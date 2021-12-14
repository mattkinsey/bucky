"""Wrapper for an async thread operating on data in a queue"""
import queue
import threading


def _thread_target(_queue, func, pre_func, post_func, **kwargs):
    """Wrapper around functionals that becomes the target for the thread"""
    if pre_func is not None:
        pre_func_output = pre_func(**kwargs)
    for item in iter(_queue.get, None):
        func_output = func(item, **pre_func_output, **kwargs)
        _queue.task_done()

    if post_func is not None:
        post_func(**pre_func_output, **func_output, **kwargs)
    _queue.task_done()


class AsyncQueueThread:
    """Async thread that processes data put into its queue"""

    def __init__(self, func, pre_func=None, post_func=None, queue_maxsize=100, **kwargs):
        """Init TODO describe functionals"""
        self._func = func
        self._pre_func = pre_func
        self._post_func = post_func
        self._queue = queue.Queue(queue_maxsize)
        self._thread = threading.Thread(
            target=_thread_target, args=(self._queue, self._func, self._pre_func, self._post_func), kwargs=kwargs
        )
        self._thread.start()

    def put(self, x):
        """Add item to thread's queue"""
        self._queue.put(x)

    def close(self):
        """Close thread and clean up"""
        self._queue.put(None)
        self._thread.join()


if __name__ == "__main__":

    def func(x, asdf, y, **kwargs):
        """test target"""
        print(asdf)
        print(x + y)

    def pre_func(**kwargs):
        """test pre_func"""
        y = 2
        print(locals())
        return {"y": y}

    test = AsyncQueueThread(func, pre_func=pre_func, asdf="asdf")
    for i in range(10):
        test.put(i)

    test.close()

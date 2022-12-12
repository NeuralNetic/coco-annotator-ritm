import threading
from typing import List
from threading import Lock
import queue


class FunctionServingWrapper(object):
    """
    Class of wrapper for restriction count of simultaneous function calls
    """

    def __init__(self,
                 callable_functions: List[callable]):
        self.callable_functions = callable_functions
        self.callable_functions_threads = None
        self.n_workers = len(self.callable_functions)
        self.work_q = queue.Queue()
        self.res_q = queue.Queue()
        self.index_mutex = Lock()
        self.current_index = 0
        self.is_stop = False

    def start(self):
        self.is_stop = False
        self.callable_functions_threads = [
            threading.Thread(target=self.worker_function, args=(fc,))
            for fc in self.callable_functions
        ]
        for ft in self.callable_functions_threads:
            ft.start()

        return self

    def stop(self):
        self.is_stop = True
        assert self.callable_functions_threads is not None
        for ft in self.callable_functions_threads:
            ft.join()

    def worker_function(self, func: callable):
        while True:
            if self.is_stop:
                break

            try:
                _args, task_id = self.work_q.get(block=False)
                self.work_q.task_done()
            except queue.Empty:
                continue
            else:
                result = func(*_args)
                self.res_q.put([result, task_id])

    def get_id(self):
        with self.index_mutex:
            _id = self.current_index
            self.current_index += 1
            if self.current_index > 1000000:
                self.current_index = 0
        return _id

    def __call__(self, *_args):
        """
        Run call method of target callable function
        Args:
            *_args: args for callable function
        Returns:
            Return callable function results
        """
        task_id = self.get_id()
        self.work_q.put([_args, task_id])

        result = None
        while True:
            if self.is_stop:
                break
            try:
                if self.res_q.queue[0][1] == task_id:
                    result = self.res_q.get(block=False)[0]
                    self.res_q.task_done()
                    break
            except IndexError:
                continue

        return result

    def __del__(self):
        self.stop()

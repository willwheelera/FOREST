import time

class Timer:
    def __init__(self, fill=0, mute=False):
        self.t0 = time.perf_counter()
        self.fill = fill
        self.print = self._print if not mute else self._pass

    def _print(self, s):
        t = time.perf_counter() - self.t0
        print(s.ljust(self.fill), round(t, 2), flush=True)

    def _pass(self, s):
        pass

    def reset(self):
        self.t0 = time.perf_counter()


import time

class Timer:
    def __init__(self, fill=0, round_to=2, mute=False):
        self.t0 = time.perf_counter()
        self.fill = fill
        self.print = self._print if not mute else self._pass
        self.round_to = round_to

    def _print(self, s, **kws):
        t = time.perf_counter() - self.t0
        print(s.ljust(self.fill), round(t, self.round_to), flush=True, **kws)

    def _pass(self, s, **kws):
        pass

    def reset(self):
        self.t0 = time.perf_counter()


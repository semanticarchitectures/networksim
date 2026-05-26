"""Simulation clock with configurable step size and partial final step handling."""


class SimulationClock:
    """
    Deterministic simulation clock that advances in discrete time steps.

    The clock starts at a configured start time and advances by step_size
    until it reaches the configured end time. When the remaining time is
    less than step_size, the clock advances by the remaining time to reach
    exactly the end time (partial final step).

    Validates at construction that step_size > 0 and step_size <= duration.
    """

    def __init__(self, start: float, end: float, step_size: float) -> None:
        """
        Initialize the simulation clock.

        Args:
            start: Simulation start time in seconds.
            end: Simulation end time in seconds.
            step_size: Time step size in seconds (0.001 to 3600.0).

        Raises:
            ValueError: If step_size <= 0 or step_size > duration.
        """
        duration = end - start

        if step_size <= 0:
            raise ValueError(
                f"step_size must be greater than 0, got {step_size}. "
                f"Acceptable range: 0.001 to 3600.0 seconds."
            )

        if step_size > duration:
            raise ValueError(
                f"step_size ({step_size}) exceeds duration ({duration}). "
                f"step_size must be <= duration."
            )

        self._start = start
        self._end = end
        self._step_size = step_size
        self._current_time = start

    @property
    def current_time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._current_time

    @property
    def step_size(self) -> float:
        """Return the configured step size in seconds."""
        return self._step_size

    def is_finished(self) -> bool:
        """Return True if the simulation clock has reached or exceeded the end time."""
        return self._current_time >= self._end

    def advance(self) -> None:
        """
        Advance the clock by one step.

        If the remaining time is less than step_size, advances by the
        remaining time to reach exactly the end time (partial final step).
        Does nothing if the clock is already finished.
        """
        if self.is_finished():
            return

        remaining = self._end - self._current_time
        if remaining <= self._step_size:
            # Partial final step: advance to exactly end time
            self._current_time = self._end
        else:
            self._current_time += self._step_size

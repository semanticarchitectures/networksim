"""Unit tests for SimulationClock."""

import pytest
from manet_sim.core.clock import SimulationClock


class TestSimulationClockInit:
    """Tests for SimulationClock construction and validation."""

    def test_valid_construction(self):
        """Clock initializes with valid parameters."""
        clock = SimulationClock(start=0.0, end=3600.0, step_size=1.0)
        assert clock.current_time == 0.0
        assert clock.step_size == 1.0
        assert not clock.is_finished()

    def test_step_size_zero_raises(self):
        """step_size of 0 raises ValueError."""
        with pytest.raises(ValueError, match="step_size must be greater than 0"):
            SimulationClock(start=0.0, end=3600.0, step_size=0.0)

    def test_step_size_negative_raises(self):
        """Negative step_size raises ValueError."""
        with pytest.raises(ValueError, match="step_size must be greater than 0"):
            SimulationClock(start=0.0, end=3600.0, step_size=-1.0)

    def test_step_size_exceeds_duration_raises(self):
        """step_size greater than duration raises ValueError."""
        with pytest.raises(ValueError, match="exceeds duration"):
            SimulationClock(start=0.0, end=10.0, step_size=11.0)

    def test_step_size_equals_duration_valid(self):
        """step_size equal to duration is valid (single step simulation)."""
        clock = SimulationClock(start=0.0, end=10.0, step_size=10.0)
        assert not clock.is_finished()

    def test_minimum_step_size(self):
        """Minimum step_size of 0.001 is accepted."""
        clock = SimulationClock(start=0.0, end=1.0, step_size=0.001)
        assert clock.step_size == 0.001

    def test_non_zero_start_time(self):
        """Clock works with non-zero start time."""
        clock = SimulationClock(start=10.0, end=20.0, step_size=2.0)
        assert clock.current_time == 10.0


class TestSimulationClockAdvance:
    """Tests for clock advancement behavior."""

    def test_advance_single_step(self):
        """Clock advances by step_size."""
        clock = SimulationClock(start=0.0, end=10.0, step_size=2.0)
        clock.advance()
        assert clock.current_time == 2.0

    def test_advance_to_completion(self):
        """Clock advances to exactly end time when evenly divisible."""
        clock = SimulationClock(start=0.0, end=10.0, step_size=2.0)
        steps = 0
        while not clock.is_finished():
            clock.advance()
            steps += 1
        assert clock.current_time == 10.0
        assert steps == 5

    def test_partial_final_step(self):
        """Clock handles partial final step correctly."""
        # 10 / 3 = 3 full steps + 1 partial step of 1.0
        clock = SimulationClock(start=0.0, end=10.0, step_size=3.0)
        clock.advance()  # 3.0
        clock.advance()  # 6.0
        clock.advance()  # 9.0
        assert clock.current_time == 9.0
        assert not clock.is_finished()
        clock.advance()  # 10.0 (partial step of 1.0)
        assert clock.current_time == 10.0
        assert clock.is_finished()

    def test_advance_after_finished_does_nothing(self):
        """Advancing after completion does not change time."""
        clock = SimulationClock(start=0.0, end=2.0, step_size=1.0)
        clock.advance()  # 1.0
        clock.advance()  # 2.0
        assert clock.is_finished()
        clock.advance()  # Should do nothing
        assert clock.current_time == 2.0

    def test_single_step_simulation(self):
        """step_size equals duration results in one step."""
        clock = SimulationClock(start=0.0, end=5.0, step_size=5.0)
        assert not clock.is_finished()
        clock.advance()
        assert clock.current_time == 5.0
        assert clock.is_finished()

    def test_exact_end_time_with_fractional_steps(self):
        """Clock reaches exactly end time with fractional step sizes."""
        clock = SimulationClock(start=0.0, end=1.0, step_size=0.3)
        while not clock.is_finished():
            clock.advance()
        assert clock.current_time == 1.0

    def test_large_duration_small_step(self):
        """Clock works with large duration and small step size."""
        clock = SimulationClock(start=0.0, end=3600.0, step_size=1.0)
        # Advance a few steps and verify
        for _ in range(100):
            clock.advance()
        assert clock.current_time == 100.0

    def test_is_finished_at_exact_end(self):
        """is_finished returns True when current_time equals end time."""
        clock = SimulationClock(start=0.0, end=4.0, step_size=2.0)
        clock.advance()  # 2.0
        assert not clock.is_finished()
        clock.advance()  # 4.0
        assert clock.is_finished()

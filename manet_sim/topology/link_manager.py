"""Link state management with hysteresis for MANET topology.

Implements hysteresis-based link formation and teardown to prevent
rapid link flapping when nodes hover near the radio range boundary.
"""

from enum import Enum, auto


class LinkState(Enum):
    """State of a link between two nodes."""

    ABSENT = auto()
    ACTIVE = auto()


class LinkManager:
    """Manages link state transitions with hysteresis buffer zone.

    The hysteresis mechanism defines two thresholds around the radio range:
    - inner_threshold = radio_range - margin (where margin = radio_range * hysteresis_pct)
    - outer_threshold = radio_range + margin

    State transitions:
    - ABSENT → ACTIVE: when distance < inner_threshold
    - ACTIVE → ABSENT: when distance > outer_threshold
    - In hysteresis zone (between inner and outer): state unchanged
    """

    def __init__(self, radio_range: float, hysteresis_pct: float = 0.10) -> None:
        """Initialize LinkManager.

        Args:
            radio_range: Radio transmission range in miles.
            hysteresis_pct: Hysteresis margin as a fraction of radio_range (default 10%).
        """
        if radio_range <= 0:
            raise ValueError("radio_range must be positive")
        if hysteresis_pct < 0:
            raise ValueError("hysteresis_pct must be non-negative")

        self._radio_range = radio_range
        self._hysteresis_pct = hysteresis_pct
        self._margin = radio_range * hysteresis_pct

    @property
    def radio_range(self) -> float:
        """The configured radio transmission range."""
        return self._radio_range

    @property
    def hysteresis_pct(self) -> float:
        """The hysteresis margin percentage."""
        return self._hysteresis_pct

    @property
    def inner_threshold(self) -> float:
        """Distance below which ABSENT links become ACTIVE."""
        return self._radio_range - self._margin

    @property
    def outer_threshold(self) -> float:
        """Distance above which ACTIVE links become ABSENT."""
        return self._radio_range + self._margin

    def evaluate(self, current_state: LinkState, distance: float) -> LinkState:
        """Evaluate link state transition based on current state and distance.

        Args:
            current_state: The current link state (ABSENT or ACTIVE).
            distance: Current Euclidean distance between the two nodes.

        Returns:
            The new link state after applying hysteresis rules.
        """
        if current_state == LinkState.ABSENT:
            if distance < self.inner_threshold:
                return LinkState.ACTIVE
            return LinkState.ABSENT

        elif current_state == LinkState.ACTIVE:
            if distance > self.outer_threshold:
                return LinkState.ABSENT
            return LinkState.ACTIVE

        # Fallback (should not be reached with the two-state enum)
        return current_state

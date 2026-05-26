"""Unit tests for LinkManager with hysteresis-based link state transitions."""

import pytest

from manet_sim.topology.link_manager import LinkManager, LinkState


class TestLinkManagerInit:
    """Tests for LinkManager initialization."""

    def test_basic_initialization(self):
        lm = LinkManager(radio_range=1.0, hysteresis_pct=0.10)
        assert lm.radio_range == 1.0
        assert lm.hysteresis_pct == 0.10

    def test_default_hysteresis_pct(self):
        lm = LinkManager(radio_range=5.0)
        assert lm.hysteresis_pct == 0.10

    def test_custom_hysteresis_pct(self):
        lm = LinkManager(radio_range=10.0, hysteresis_pct=0.20)
        assert lm.hysteresis_pct == 0.20

    def test_zero_hysteresis_pct(self):
        """Zero hysteresis means inner == outer == radio_range."""
        lm = LinkManager(radio_range=1.0, hysteresis_pct=0.0)
        assert lm.inner_threshold == 1.0
        assert lm.outer_threshold == 1.0

    def test_invalid_radio_range_zero(self):
        with pytest.raises(ValueError, match="radio_range must be positive"):
            LinkManager(radio_range=0.0)

    def test_invalid_radio_range_negative(self):
        with pytest.raises(ValueError, match="radio_range must be positive"):
            LinkManager(radio_range=-1.0)

    def test_invalid_hysteresis_pct_negative(self):
        with pytest.raises(ValueError, match="hysteresis_pct must be non-negative"):
            LinkManager(radio_range=1.0, hysteresis_pct=-0.1)


class TestThresholds:
    """Tests for inner and outer threshold calculations."""

    def test_thresholds_default(self):
        """With radio_range=1.0 and 10% hysteresis: inner=0.9, outer=1.1."""
        lm = LinkManager(radio_range=1.0, hysteresis_pct=0.10)
        assert lm.inner_threshold == pytest.approx(0.9)
        assert lm.outer_threshold == pytest.approx(1.1)

    def test_thresholds_large_range(self):
        """With radio_range=10.0 and 10% hysteresis: inner=9.0, outer=11.0."""
        lm = LinkManager(radio_range=10.0, hysteresis_pct=0.10)
        assert lm.inner_threshold == pytest.approx(9.0)
        assert lm.outer_threshold == pytest.approx(11.0)

    def test_thresholds_large_hysteresis(self):
        """With radio_range=5.0 and 20% hysteresis: inner=4.0, outer=6.0."""
        lm = LinkManager(radio_range=5.0, hysteresis_pct=0.20)
        assert lm.inner_threshold == pytest.approx(4.0)
        assert lm.outer_threshold == pytest.approx(6.0)

    def test_thresholds_zero_hysteresis(self):
        """With 0% hysteresis, inner == outer == radio_range."""
        lm = LinkManager(radio_range=3.0, hysteresis_pct=0.0)
        assert lm.inner_threshold == pytest.approx(3.0)
        assert lm.outer_threshold == pytest.approx(3.0)


class TestEvaluateAbsentState:
    """Tests for evaluate() when current state is ABSENT."""

    def setup_method(self):
        self.lm = LinkManager(radio_range=1.0, hysteresis_pct=0.10)
        # inner_threshold = 0.9, outer_threshold = 1.1

    def test_absent_distance_below_inner_becomes_active(self):
        """ABSENT → ACTIVE when distance < inner_threshold."""
        result = self.lm.evaluate(LinkState.ABSENT, 0.5)
        assert result == LinkState.ACTIVE

    def test_absent_distance_just_below_inner_becomes_active(self):
        """ABSENT → ACTIVE when distance is just below inner_threshold."""
        result = self.lm.evaluate(LinkState.ABSENT, 0.89)
        assert result == LinkState.ACTIVE

    def test_absent_distance_at_inner_stays_absent(self):
        """ABSENT stays ABSENT when distance == inner_threshold (not strictly less)."""
        result = self.lm.evaluate(LinkState.ABSENT, 0.9)
        assert result == LinkState.ABSENT

    def test_absent_distance_in_hysteresis_zone_stays_absent(self):
        """ABSENT stays ABSENT in hysteresis zone (Req 12.6)."""
        result = self.lm.evaluate(LinkState.ABSENT, 0.95)
        assert result == LinkState.ABSENT

    def test_absent_distance_at_radio_range_stays_absent(self):
        """ABSENT stays ABSENT at exactly radio_range."""
        result = self.lm.evaluate(LinkState.ABSENT, 1.0)
        assert result == LinkState.ABSENT

    def test_absent_distance_above_outer_stays_absent(self):
        """ABSENT stays ABSENT when distance > outer_threshold."""
        result = self.lm.evaluate(LinkState.ABSENT, 2.0)
        assert result == LinkState.ABSENT

    def test_absent_distance_zero_becomes_active(self):
        """ABSENT → ACTIVE when distance is 0 (same position)."""
        result = self.lm.evaluate(LinkState.ABSENT, 0.0)
        assert result == LinkState.ACTIVE


class TestEvaluateActiveState:
    """Tests for evaluate() when current state is ACTIVE."""

    def setup_method(self):
        self.lm = LinkManager(radio_range=1.0, hysteresis_pct=0.10)
        # inner_threshold = 0.9, outer_threshold = 1.1

    def test_active_distance_below_inner_stays_active(self):
        """ACTIVE stays ACTIVE when distance < inner_threshold."""
        result = self.lm.evaluate(LinkState.ACTIVE, 0.5)
        assert result == LinkState.ACTIVE

    def test_active_distance_in_hysteresis_zone_stays_active(self):
        """ACTIVE stays ACTIVE in hysteresis zone (Req 12.3)."""
        result = self.lm.evaluate(LinkState.ACTIVE, 0.95)
        assert result == LinkState.ACTIVE

    def test_active_distance_at_radio_range_stays_active(self):
        """ACTIVE stays ACTIVE at exactly radio_range."""
        result = self.lm.evaluate(LinkState.ACTIVE, 1.0)
        assert result == LinkState.ACTIVE

    def test_active_distance_at_outer_stays_active(self):
        """ACTIVE stays ACTIVE when distance == outer_threshold (not strictly greater)."""
        result = self.lm.evaluate(LinkState.ACTIVE, 1.1)
        assert result == LinkState.ACTIVE

    def test_active_distance_above_outer_becomes_absent(self):
        """ACTIVE → ABSENT when distance > outer_threshold."""
        result = self.lm.evaluate(LinkState.ACTIVE, 1.2)
        assert result == LinkState.ABSENT

    def test_active_distance_far_above_outer_becomes_absent(self):
        """ACTIVE → ABSENT when distance is far above outer_threshold."""
        result = self.lm.evaluate(LinkState.ACTIVE, 5.0)
        assert result == LinkState.ABSENT

    def test_active_distance_just_above_outer_becomes_absent(self):
        """ACTIVE → ABSENT when distance is just above outer_threshold."""
        result = self.lm.evaluate(LinkState.ACTIVE, 1.1001)
        assert result == LinkState.ABSENT


class TestHysteresisZoneBehavior:
    """Tests verifying hysteresis zone prevents link flapping."""

    def setup_method(self):
        self.lm = LinkManager(radio_range=1.0, hysteresis_pct=0.10)

    def test_absent_pair_entering_buffer_zone_stays_absent(self):
        """Req 12.6: Previously unlinked pair in buffer zone stays ABSENT."""
        # Node pair starts far away (ABSENT), moves into hysteresis zone
        state = LinkState.ABSENT
        # Distance in hysteresis zone: between 0.9 and 1.1
        state = self.lm.evaluate(state, 0.95)
        assert state == LinkState.ABSENT
        state = self.lm.evaluate(state, 0.92)
        assert state == LinkState.ABSENT
        state = self.lm.evaluate(state, 1.05)
        assert state == LinkState.ABSENT

    def test_active_pair_in_buffer_zone_stays_active(self):
        """Req 12.3: Active link in buffer zone stays ACTIVE."""
        # Node pair starts close (ACTIVE), moves into hysteresis zone
        state = LinkState.ACTIVE
        state = self.lm.evaluate(state, 0.95)
        assert state == LinkState.ACTIVE
        state = self.lm.evaluate(state, 1.0)
        assert state == LinkState.ACTIVE
        state = self.lm.evaluate(state, 1.05)
        assert state == LinkState.ACTIVE

    def test_full_lifecycle_form_and_break(self):
        """Test complete link lifecycle: form → hold → break."""
        state = LinkState.ABSENT

        # Nodes approach: still ABSENT in buffer zone
        state = self.lm.evaluate(state, 0.95)
        assert state == LinkState.ABSENT

        # Nodes get close enough: ACTIVE
        state = self.lm.evaluate(state, 0.8)
        assert state == LinkState.ACTIVE

        # Nodes drift into buffer zone: still ACTIVE
        state = self.lm.evaluate(state, 0.95)
        assert state == LinkState.ACTIVE

        state = self.lm.evaluate(state, 1.05)
        assert state == LinkState.ACTIVE

        # Nodes move beyond outer threshold: ABSENT
        state = self.lm.evaluate(state, 1.2)
        assert state == LinkState.ABSENT

    def test_oscillation_in_buffer_zone_no_flapping(self):
        """Nodes oscillating in buffer zone should not cause state changes."""
        # Start as ACTIVE (formed earlier)
        state = LinkState.ACTIVE
        distances = [0.92, 0.98, 1.05, 0.95, 1.08, 0.91, 1.0]
        for d in distances:
            state = self.lm.evaluate(state, d)
            assert state == LinkState.ACTIVE

        # Start as ABSENT (never formed)
        state = LinkState.ABSENT
        for d in distances:
            state = self.lm.evaluate(state, d)
            assert state == LinkState.ABSENT


class TestDifferentConfigurations:
    """Tests with various radio_range and hysteresis_pct values."""

    def test_large_radio_range(self):
        """Test with radio_range=10.0, hysteresis=10%: inner=9.0, outer=11.0."""
        lm = LinkManager(radio_range=10.0, hysteresis_pct=0.10)
        assert lm.evaluate(LinkState.ABSENT, 8.5) == LinkState.ACTIVE
        assert lm.evaluate(LinkState.ABSENT, 9.5) == LinkState.ABSENT
        assert lm.evaluate(LinkState.ACTIVE, 10.5) == LinkState.ACTIVE
        assert lm.evaluate(LinkState.ACTIVE, 11.5) == LinkState.ABSENT

    def test_small_radio_range(self):
        """Test with radio_range=0.5, hysteresis=10%: inner=0.45, outer=0.55."""
        lm = LinkManager(radio_range=0.5, hysteresis_pct=0.10)
        assert lm.evaluate(LinkState.ABSENT, 0.4) == LinkState.ACTIVE
        assert lm.evaluate(LinkState.ABSENT, 0.5) == LinkState.ABSENT
        assert lm.evaluate(LinkState.ACTIVE, 0.5) == LinkState.ACTIVE
        assert lm.evaluate(LinkState.ACTIVE, 0.6) == LinkState.ABSENT

    def test_large_hysteresis(self):
        """Test with radio_range=1.0, hysteresis=50%: inner=0.5, outer=1.5."""
        lm = LinkManager(radio_range=1.0, hysteresis_pct=0.50)
        assert lm.evaluate(LinkState.ABSENT, 0.4) == LinkState.ACTIVE
        assert lm.evaluate(LinkState.ABSENT, 0.6) == LinkState.ABSENT
        assert lm.evaluate(LinkState.ACTIVE, 1.4) == LinkState.ACTIVE
        assert lm.evaluate(LinkState.ACTIVE, 1.6) == LinkState.ABSENT

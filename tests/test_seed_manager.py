"""Unit tests for SeedManager."""

import numpy as np
import pytest

from manet_sim.core.seed_manager import SeedManager


class TestSeedManagerInit:
    """Tests for SeedManager initialization and seed validation."""

    def test_explicit_seed_stored(self):
        """Explicit seed is stored and accessible via property."""
        sm = SeedManager(seed=42)
        assert sm.seed == 42

    def test_explicit_seed_zero(self):
        """Seed of 0 is valid."""
        sm = SeedManager(seed=0)
        assert sm.seed == 0

    def test_explicit_seed_max(self):
        """Maximum seed value (2^32 - 1) is valid."""
        sm = SeedManager(seed=2**32 - 1)
        assert sm.seed == 2**32 - 1

    def test_auto_generated_seed_when_none(self):
        """When no seed provided, one is auto-generated."""
        sm = SeedManager(seed=None)
        assert isinstance(sm.seed, int)
        assert 0 <= sm.seed <= 2**32 - 1

    def test_auto_generated_seed_default(self):
        """Default constructor auto-generates a seed."""
        sm = SeedManager()
        assert isinstance(sm.seed, int)
        assert 0 <= sm.seed <= 2**32 - 1

    def test_negative_seed_rejected(self):
        """Negative seed raises ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            SeedManager(seed=-1)

    def test_seed_too_large_rejected(self):
        """Seed > 2^32-1 raises ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            SeedManager(seed=2**32)

    def test_float_seed_rejected(self):
        """Float seed raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            SeedManager(seed=3.14)  # type: ignore

    def test_string_seed_rejected(self):
        """String seed raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            SeedManager(seed="42")  # type: ignore


class TestSeedManagerGetRng:
    """Tests for get_rng method."""

    def test_returns_generator(self):
        """get_rng returns a numpy Generator instance."""
        sm = SeedManager(seed=42)
        rng = sm.get_rng("mobility")
        assert isinstance(rng, np.random.Generator)

    def test_same_subsystem_returns_same_instance(self):
        """Calling get_rng with same name returns the same Generator."""
        sm = SeedManager(seed=42)
        rng1 = sm.get_rng("mobility")
        rng2 = sm.get_rng("mobility")
        assert rng1 is rng2

    def test_different_subsystems_return_different_instances(self):
        """Different subsystem names produce different Generator instances."""
        sm = SeedManager(seed=42)
        rng_mobility = sm.get_rng("mobility")
        rng_topology = sm.get_rng("topology")
        assert rng_mobility is not rng_topology

    def test_different_subsystems_produce_different_sequences(self):
        """Different subsystems produce different random sequences."""
        sm = SeedManager(seed=42)
        rng_a = sm.get_rng("mobility")
        rng_b = sm.get_rng("topology")
        vals_a = rng_a.random(10)
        vals_b = rng_b.random(10)
        assert not np.array_equal(vals_a, vals_b)

    def test_empty_subsystem_rejected(self):
        """Empty string subsystem raises ValueError."""
        sm = SeedManager(seed=42)
        with pytest.raises(ValueError, match="non-empty string"):
            sm.get_rng("")

    def test_non_string_subsystem_rejected(self):
        """Non-string subsystem raises ValueError."""
        sm = SeedManager(seed=42)
        with pytest.raises(ValueError, match="non-empty string"):
            sm.get_rng(123)  # type: ignore


class TestSeedManagerReproducibility:
    """Tests for reproducibility guarantees."""

    def test_same_seed_same_sequence(self):
        """Same seed produces identical RNG sequences."""
        sm1 = SeedManager(seed=99)
        sm2 = SeedManager(seed=99)
        vals1 = sm1.get_rng("mobility").random(100)
        vals2 = sm2.get_rng("mobility").random(100)
        np.testing.assert_array_equal(vals1, vals2)

    def test_different_seeds_different_sequences(self):
        """Different seeds produce different RNG sequences."""
        sm1 = SeedManager(seed=1)
        sm2 = SeedManager(seed=2)
        vals1 = sm1.get_rng("mobility").random(100)
        vals2 = sm2.get_rng("mobility").random(100)
        assert not np.array_equal(vals1, vals2)

    def test_order_independence(self):
        """Requesting subsystems in different order produces same sequences."""
        sm1 = SeedManager(seed=42)
        sm2 = SeedManager(seed=42)

        # Request in different order
        rng1_mob = sm1.get_rng("mobility")
        rng1_top = sm1.get_rng("topology")

        rng2_top = sm2.get_rng("topology")
        rng2_mob = sm2.get_rng("mobility")

        # Same subsystem should produce same sequence regardless of request order
        vals1_mob = rng1_mob.random(50)
        vals2_mob = rng2_mob.random(50)
        np.testing.assert_array_equal(vals1_mob, vals2_mob)

        vals1_top = rng1_top.random(50)
        vals2_top = rng2_top.random(50)
        np.testing.assert_array_equal(vals1_top, vals2_top)

    def test_subsystem_independence(self):
        """Using one subsystem's RNG doesn't affect another's sequence."""
        sm1 = SeedManager(seed=42)
        sm2 = SeedManager(seed=42)

        # In sm1, use mobility RNG extensively before getting topology
        rng1_mob = sm1.get_rng("mobility")
        rng1_mob.random(1000)  # Advance mobility RNG state

        # In sm2, get topology directly
        rng1_top = sm1.get_rng("topology")
        rng2_top = sm2.get_rng("topology")

        # Topology sequences should still be identical
        vals1 = rng1_top.random(50)
        vals2 = rng2_top.random(50)
        np.testing.assert_array_equal(vals1, vals2)

    def test_multiple_subsystems(self):
        """Multiple subsystems all produce reproducible, independent streams."""
        subsystems = ["mobility", "topology", "placement", "network", "output"]
        sm1 = SeedManager(seed=123)
        sm2 = SeedManager(seed=123)

        for name in subsystems:
            vals1 = sm1.get_rng(name).random(20)
            vals2 = sm2.get_rng(name).random(20)
            np.testing.assert_array_equal(vals1, vals2)

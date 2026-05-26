"""Property-based tests for SeedManager reproducibility.

Feature: manet-simulation-engine, Property 4: Simulation reproducibility
**Validates: Requirements 4.1, 4.2, 4.3**

Tests that for any valid seed (0 to 2^32-1) and valid configuration,
creating two SeedManagers with the same seed SHALL produce bit-for-bit
identical RNG sequences for the same subsystem names, regardless of the
order subsystems are requested.
"""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from manet_sim.core.seed_manager import SeedManager


# --- Strategies ---

# Valid seed range: 0 to 2^32 - 1
valid_seeds = st.integers(min_value=0, max_value=2**32 - 1)

# Subsystem names: non-empty strings of reasonable length
subsystem_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=50,
)

# Lists of unique subsystem names (at least 2 for order-independence testing)
subsystem_lists = st.lists(
    subsystem_names,
    min_size=2,
    max_size=10,
    unique=True,
)

# Number of random values to generate for comparison
sequence_lengths = st.integers(min_value=1, max_value=200)


# --- Property Tests ---


class TestSeedManagerReproducibilityProperty:
    """Property 4: Simulation reproducibility.

    Same seed produces identical RNG sequences across subsystems.
    """

    @given(seed=valid_seeds, subsystem=subsystem_names, n=sequence_lengths)
    @settings(max_examples=100)
    def test_same_seed_produces_identical_sequences(
        self, seed: int, subsystem: str, n: int
    ):
        """For any valid seed and subsystem name, two SeedManagers with the
        same seed produce bit-for-bit identical RNG sequences."""
        sm1 = SeedManager(seed=seed)
        sm2 = SeedManager(seed=seed)

        rng1 = sm1.get_rng(subsystem)
        rng2 = sm2.get_rng(subsystem)

        vals1 = rng1.random(n)
        vals2 = rng2.random(n)

        np.testing.assert_array_equal(vals1, vals2)

    @given(seed=valid_seeds, subsystems=subsystem_lists, n=sequence_lengths)
    @settings(max_examples=100)
    def test_order_independence(
        self, seed: int, subsystems: list, n: int
    ):
        """For any valid seed and set of subsystem names, requesting subsystems
        in different orders produces identical sequences for each subsystem."""
        sm1 = SeedManager(seed=seed)
        sm2 = SeedManager(seed=seed)

        # Request subsystems in original order for sm1
        for name in subsystems:
            sm1.get_rng(name)

        # Request subsystems in reversed order for sm2
        for name in reversed(subsystems):
            sm2.get_rng(name)

        # Verify each subsystem produces identical sequences
        for name in subsystems:
            vals1 = sm1.get_rng(name).random(n)
            vals2 = sm2.get_rng(name).random(n)
            np.testing.assert_array_equal(
                vals1,
                vals2,
                err_msg=f"Subsystem '{name}' produced different sequences "
                f"depending on request order (seed={seed})",
            )

    @given(
        seed=valid_seeds,
        subsystems=subsystem_lists,
        advance_counts=st.lists(
            st.integers(min_value=0, max_value=500), min_size=2, max_size=10
        ),
        n=sequence_lengths,
    )
    @settings(max_examples=100)
    def test_subsystem_independence(
        self, seed: int, subsystems: list, advance_counts: list, n: int
    ):
        """Advancing one subsystem's RNG does not affect another subsystem's
        sequence. Each subsystem stream is fully independent."""
        # Ensure we have enough advance_counts for our subsystems
        assume(len(advance_counts) >= len(subsystems))

        sm1 = SeedManager(seed=seed)
        sm2 = SeedManager(seed=seed)

        # In sm1, advance each subsystem's RNG by different amounts
        for i, name in enumerate(subsystems):
            rng = sm1.get_rng(name)
            rng.random(advance_counts[i])  # Advance state

        # In sm2, don't advance any RNG — just get them fresh
        # But since get_rng returns cached instances, we need a fresh manager
        # to compare the "target" subsystem
        target_subsystem = subsystems[-1]

        # Create a third manager to get the expected sequence for target
        sm3 = SeedManager(seed=seed)
        rng3 = sm3.get_rng(target_subsystem)

        # Advance sm3's target by the same amount as sm1's target
        target_advance = advance_counts[len(subsystems) - 1]
        rng3.random(target_advance)

        # Now the next n values from sm1's target should match sm3's target
        rng1_target = sm1.get_rng(target_subsystem)
        vals1 = rng1_target.random(n)
        vals3 = rng3.random(n)

        np.testing.assert_array_equal(
            vals1,
            vals3,
            err_msg=f"Subsystem '{target_subsystem}' sequence was affected by "
            f"other subsystems' usage (seed={seed})",
        )

    @given(
        seed1=valid_seeds,
        seed2=valid_seeds,
        subsystem=subsystem_names,
        n=sequence_lengths,
    )
    @settings(max_examples=100)
    def test_different_seeds_produce_different_sequences(
        self, seed1: int, seed2: int, subsystem: str, n: int
    ):
        """Different seeds produce different RNG sequences for the same
        subsystem (with overwhelming probability for non-trivial n)."""
        assume(seed1 != seed2)
        assume(n >= 10)  # Need enough values to avoid accidental collisions

        sm1 = SeedManager(seed=seed1)
        sm2 = SeedManager(seed=seed2)

        vals1 = sm1.get_rng(subsystem).random(n)
        vals2 = sm2.get_rng(subsystem).random(n)

        # Different seeds should produce different sequences
        assert not np.array_equal(vals1, vals2)

"""Seed management for reproducible simulation runs.

Provides centralized RNG seeding using NumPy's SeedSequence to create
independent Generator instances for each subsystem (mobility, topology,
placement, etc.) so they don't interfere with each other.
"""

import hashlib
from typing import Optional

import numpy as np


class SeedManager:
    """Manages random number generation seeding for simulation reproducibility.

    Uses numpy's SeedSequence to create independent Generator instances
    for each subsystem, ensuring that subsystems don't interfere with
    each other's random streams while maintaining full reproducibility.

    Each subsystem name is hashed to produce a deterministic spawn key,
    so the order in which subsystems request their RNG does not affect
    the sequences produced.

    Args:
        seed: Non-negative integer seed (0 to 2^32-1). If None, a seed
              is auto-generated and recorded for reproducibility.

    Raises:
        ValueError: If seed is outside the valid range or not an integer.
    """

    MAX_SEED = 2**32 - 1

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            # Generate a random seed and record it
            self._seed = int(np.random.SeedSequence().entropy % (self.MAX_SEED + 1))
        else:
            self._validate_seed(seed)
            self._seed = seed

        # Create the root SeedSequence for deriving child sequences
        self._seed_sequence = np.random.SeedSequence(self._seed)

        # Cache of subsystem RNGs keyed by subsystem name
        self._rng_cache: dict[str, np.random.Generator] = {}

    def _validate_seed(self, seed: int) -> None:
        """Validate that seed is a non-negative integer within range."""
        if not isinstance(seed, int):
            raise ValueError(
                f"Seed must be a non-negative integer, got {type(seed).__name__}: {seed}"
            )
        if seed < 0 or seed > self.MAX_SEED:
            raise ValueError(
                f"Seed must be in range [0, {self.MAX_SEED}], got {seed}"
            )

    @property
    def seed(self) -> int:
        """The seed used for this simulation run (generated if not provided)."""
        return self._seed

    def get_rng(self, subsystem: str) -> np.random.Generator:
        """Get an independent RNG for a named subsystem.

        Each subsystem gets its own Generator instance derived from the
        root SeedSequence. The same subsystem name always returns the
        same Generator instance, and different subsystem names produce
        independent streams regardless of call order.

        Args:
            subsystem: Name of the subsystem (e.g., "mobility", "topology",
                      "placement"). Must be a non-empty string.

        Returns:
            A numpy Generator instance independent from other subsystems.

        Raises:
            ValueError: If subsystem is empty or not a string.
        """
        if not isinstance(subsystem, str) or not subsystem:
            raise ValueError(
                f"Subsystem must be a non-empty string, got: {subsystem!r}"
            )

        if subsystem not in self._rng_cache:
            # Derive a deterministic spawn key from the subsystem name
            # This ensures order-independence: requesting "mobility" then
            # "topology" produces the same RNGs as "topology" then "mobility"
            spawn_key = self._subsystem_to_spawn_key(subsystem)
            child_sequence = np.random.SeedSequence(
                self._seed_sequence.entropy, spawn_key=(spawn_key,)
            )
            self._rng_cache[subsystem] = np.random.default_rng(child_sequence)

        return self._rng_cache[subsystem]

    @staticmethod
    def _subsystem_to_spawn_key(subsystem: str) -> int:
        """Convert a subsystem name to a deterministic integer spawn key."""
        # Use a hash to map arbitrary subsystem names to integers
        digest = hashlib.sha256(subsystem.encode("utf-8")).digest()
        # Take first 4 bytes as an unsigned 32-bit integer
        return int.from_bytes(digest[:4], byteorder="big")

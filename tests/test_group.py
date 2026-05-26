"""Unit tests for the Group data structure."""

import numpy as np
import pytest
from collections import deque

from manet_sim.mobility.group_mobility import Group


class TestGroupCreation:
    """Tests for Group initialization and default state."""

    def test_default_group_is_empty(self):
        group = Group(group_id=1)
        assert group.member_ids == []
        assert group.leader_id is None
        assert group.is_empty
        assert group.size == 0

    def test_group_has_correct_default_arrays(self):
        group = Group(group_id=0)
        assert group.reference_point.shape == (2,)
        assert group.reference_point.dtype == np.float32
        np.testing.assert_array_equal(group.reference_point, [0.0, 0.0])
        assert group.reference_velocity.shape == (2,)
        assert group.reference_velocity.dtype == np.float32
        np.testing.assert_array_equal(group.reference_velocity, [0.0, 0.0])

    def test_group_with_custom_reference_point(self):
        ref = np.array([50.0, 75.0], dtype=np.float32)
        group = Group(group_id=5, reference_point=ref)
        np.testing.assert_array_equal(group.reference_point, [50.0, 75.0])

    def test_group_waypoint_queue_capacity(self):
        group = Group(group_id=1)
        assert group.waypoint_queue.maxlen == 100

    def test_group_initial_pause_and_speed(self):
        group = Group(group_id=1)
        assert group.pause_remaining == 0.0
        assert group.current_speed == 0.0


class TestAddMember:
    """Tests for Group.add_member()."""

    def test_add_first_member_becomes_leader(self):
        group = Group(group_id=1)
        result = group.add_member(10)
        assert result is True
        assert group.member_ids == [10]
        assert group.leader_id == 10
        assert group.size == 1

    def test_add_multiple_members_preserves_order(self):
        group = Group(group_id=1)
        group.add_member(10)
        group.add_member(20)
        group.add_member(30)
        assert group.member_ids == [10, 20, 30]
        # Leader remains the first member added
        assert group.leader_id == 10
        assert group.size == 3

    def test_add_duplicate_is_rejected(self):
        group = Group(group_id=1)
        group.add_member(10)
        group.add_member(20)
        result = group.add_member(10)
        assert result is False
        assert group.member_ids == [10, 20]
        assert group.size == 2

    def test_add_duplicate_leader_is_rejected(self):
        group = Group(group_id=1)
        group.add_member(5)
        result = group.add_member(5)
        assert result is False
        assert group.member_ids == [5]
        assert group.leader_id == 5

    def test_add_member_to_group_with_existing_leader(self):
        group = Group(group_id=1)
        group.add_member(1)
        group.add_member(2)
        # Leader should still be the first member
        assert group.leader_id == 1


class TestRemoveMember:
    """Tests for Group.remove_member()."""

    def test_remove_non_leader_member(self):
        group = Group(group_id=1)
        group.add_member(10)
        group.add_member(20)
        group.add_member(30)
        result = group.remove_member(20)
        assert result is True
        assert group.member_ids == [10, 30]
        assert group.leader_id == 10

    def test_remove_leader_promotes_longest_tenured(self):
        group = Group(group_id=1)
        group.add_member(10)  # leader, longest-tenured
        group.add_member(20)  # second longest-tenured
        group.add_member(30)
        result = group.remove_member(10)
        assert result is True
        assert group.member_ids == [20, 30]
        # 20 is now the longest-tenured remaining member
        assert group.leader_id == 20

    def test_remove_leader_with_many_members(self):
        group = Group(group_id=1)
        for i in range(1, 6):
            group.add_member(i)
        # Leader is 1, remove it
        group.remove_member(1)
        # Next longest-tenured is 2
        assert group.leader_id == 2
        assert group.member_ids == [2, 3, 4, 5]

    def test_remove_last_member_makes_group_empty(self):
        group = Group(group_id=1)
        group.add_member(42)
        result = group.remove_member(42)
        assert result is True
        assert group.member_ids == []
        assert group.leader_id is None
        assert group.is_empty

    def test_remove_nonexistent_member_returns_false(self):
        group = Group(group_id=1)
        group.add_member(10)
        result = group.remove_member(99)
        assert result is False
        assert group.member_ids == [10]
        assert group.leader_id == 10

    def test_remove_from_empty_group_returns_false(self):
        group = Group(group_id=1)
        result = group.remove_member(1)
        assert result is False

    def test_remove_non_leader_does_not_change_leader(self):
        group = Group(group_id=1)
        group.add_member(1)
        group.add_member(2)
        group.add_member(3)
        group.remove_member(3)
        assert group.leader_id == 1
        group.remove_member(2)
        assert group.leader_id == 1

    def test_successive_leader_removals(self):
        """Remove leaders one by one; each time the next longest-tenured is promoted."""
        group = Group(group_id=1)
        group.add_member(10)
        group.add_member(20)
        group.add_member(30)

        group.remove_member(10)
        assert group.leader_id == 20

        group.remove_member(20)
        assert group.leader_id == 30

        group.remove_member(30)
        assert group.leader_id is None
        assert group.is_empty


class TestGroupEdgeCases:
    """Edge case tests for Group operations."""

    def test_add_then_remove_returns_to_original(self):
        """Property 8: add then remove returns to original state."""
        group = Group(group_id=1)
        group.add_member(1)
        group.add_member(2)
        original_members = list(group.member_ids)
        original_leader = group.leader_id

        group.add_member(99)
        group.remove_member(99)

        assert group.member_ids == original_members
        assert group.leader_id == original_leader

    def test_large_group_membership(self):
        """Test group can handle up to 1000 members (requirement 7.1)."""
        group = Group(group_id=1)
        for i in range(1000):
            result = group.add_member(i)
            assert result is True
        assert group.size == 1000
        assert group.leader_id == 0  # First added

    def test_waypoint_queue_fifo_behavior(self):
        """Waypoint queue is FIFO with capacity 1-100."""
        group = Group(group_id=1)
        wp1 = np.array([10.0, 20.0], dtype=np.float32)
        wp2 = np.array([30.0, 40.0], dtype=np.float32)
        group.waypoint_queue.append(wp1)
        group.waypoint_queue.append(wp2)
        # FIFO: first in, first out
        first = group.waypoint_queue.popleft()
        np.testing.assert_array_equal(first, wp1)

    def test_waypoint_queue_respects_max_capacity(self):
        """Waypoint queue maxlen is 100; adding beyond drops oldest."""
        group = Group(group_id=1)
        for i in range(110):
            group.waypoint_queue.append(
                np.array([float(i), float(i)], dtype=np.float32)
            )
        assert len(group.waypoint_queue) == 100
        # The oldest (0-9) should have been dropped
        np.testing.assert_array_equal(
            group.waypoint_queue[0], [10.0, 10.0]
        )

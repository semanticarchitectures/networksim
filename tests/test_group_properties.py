"""Property-based tests for Group data structure.

Feature: manet-simulation-engine, Property 8: Group membership add-then-remove identity
Feature: manet-simulation-engine, Property 9: Duplicate member rejection
Feature: manet-simulation-engine, Property 10: Leader succession on removal
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from manet_sim.mobility.group_mobility import Group


# --- Strategies ---

@st.composite
def group_with_members(draw, min_members=1, max_members=50):
    """Generate a Group with a random set of unique member node IDs.

    Members are added in order, so member_ids[0] is the longest-tenured
    and is the leader.
    """
    num_members = draw(st.integers(min_value=min_members, max_value=max_members))
    # Generate unique node IDs
    member_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=10000),
            min_size=num_members,
            max_size=num_members,
            unique=True,
        )
    )
    group = Group(group_id=draw(st.integers(min_value=0, max_value=100)))
    for nid in member_ids:
        group.add_member(nid)
    return group


@st.composite
def group_and_non_member(draw):
    """Generate a Group with members and a node_id that is NOT in the group."""
    group = draw(group_with_members(min_members=1, max_members=50))
    # Pick a node_id not already in the group
    non_member = draw(st.integers(min_value=0, max_value=10000))
    assume(non_member not in group.member_ids)
    return group, non_member


@st.composite
def group_and_existing_member(draw):
    """Generate a Group with members and a node_id that IS in the group."""
    group = draw(group_with_members(min_members=1, max_members=50))
    # Pick one of the existing members
    existing_member = draw(st.sampled_from(list(group.member_ids)))
    return group, existing_member


@st.composite
def group_with_at_least_two_members(draw):
    """Generate a Group with at least 2 members for leader succession testing."""
    group = draw(group_with_members(min_members=2, max_members=50))
    return group


# --- Property Tests ---


@given(data=group_and_non_member())
@settings(max_examples=100, deadline=None)
def test_group_membership_add_then_remove_identity(data):
    """
    Property 8: Group membership add-then-remove identity.

    For any group and any node_id not already in the group, adding the node
    then removing it SHALL return the group to its original member list
    (excluding the added node).

    **Validates: Requirements 7.2, 7.4**
    """
    group, non_member = data

    # Capture original state
    original_members = list(group.member_ids)
    original_leader = group.leader_id
    original_size = group.size

    # Add the non-member
    add_result = group.add_member(non_member)
    assert add_result is True, "Adding a non-member should succeed"

    # Remove the just-added member
    remove_result = group.remove_member(non_member)
    assert remove_result is True, "Removing the just-added member should succeed"

    # Group should be back to original state
    assert group.member_ids == original_members, (
        f"Member list not restored. Expected {original_members}, got {group.member_ids}"
    )
    assert group.leader_id == original_leader, (
        f"Leader not restored. Expected {original_leader}, got {group.leader_id}"
    )
    assert group.size == original_size, (
        f"Size not restored. Expected {original_size}, got {group.size}"
    )


@given(data=group_and_existing_member())
@settings(max_examples=100, deadline=None)
def test_duplicate_member_rejection(data):
    """
    Property 9: Duplicate member rejection.

    For any group and any node_id that is already a member, attempting to add
    that node_id again SHALL be rejected and the group's member list SHALL
    remain unchanged.

    **Validates: Requirements 7.3**
    """
    group, existing_member = data

    # Capture state before duplicate add attempt
    members_before = list(group.member_ids)
    leader_before = group.leader_id
    size_before = group.size

    # Attempt to add the existing member again
    result = group.add_member(existing_member)

    # Should be rejected
    assert result is False, (
        f"Adding duplicate member {existing_member} should return False"
    )

    # Member list should be unchanged
    assert group.member_ids == members_before, (
        f"Member list changed after duplicate rejection. "
        f"Expected {members_before}, got {group.member_ids}"
    )
    assert group.leader_id == leader_before, (
        f"Leader changed after duplicate rejection. "
        f"Expected {leader_before}, got {group.leader_id}"
    )
    assert group.size == size_before, (
        f"Size changed after duplicate rejection. "
        f"Expected {size_before}, got {group.size}"
    )


@given(group=group_with_at_least_two_members())
@settings(max_examples=100, deadline=None)
def test_leader_succession_on_removal(group):
    """
    Property 10: Leader succession on removal.

    For any group with at least 2 members where the leader is removed, the new
    leader SHALL be the longest-tenured remaining member (first in the ordered
    member list after the removed leader).

    **Validates: Requirements 7.5**
    """
    # Identify the current leader
    current_leader = group.leader_id
    assert current_leader is not None, "Group with >=2 members must have a leader"
    assert current_leader == group.member_ids[0], (
        "Leader should be the first (longest-tenured) member"
    )

    # The expected new leader is the second member (next longest-tenured)
    expected_new_leader = group.member_ids[1]

    # Remove the current leader
    result = group.remove_member(current_leader)
    assert result is True, "Removing the leader should succeed"

    # The new leader should be the longest-tenured remaining member
    assert group.leader_id == expected_new_leader, (
        f"New leader should be {expected_new_leader} (longest-tenured remaining), "
        f"but got {group.leader_id}. Members: {group.member_ids}"
    )

    # The new leader should be the first in the member list
    assert group.member_ids[0] == expected_new_leader, (
        f"First member should be the new leader {expected_new_leader}, "
        f"but member_ids[0] is {group.member_ids[0]}"
    )

"""Tests for deliberate per-particle (topology) loss weights.

The key property, learned the hard way from balance_particles: reweighting must
NOT change the overall loss scale, otherwise the fixed l2_penalty silently
becomes dominant. Hence mean-1 normalization.
"""
import pytest

from spanet.network.jet_reconstruction.jet_reconstruction_base import (
    parse_particle_loss_weights,
)

PARTS = ["FRt1", "FRt2", "SRqqt1", "SRqqt2", "FBt1", "FBt2"]


def test_uniform_spec_is_identity():
    w = parse_particle_loss_weights("FRt:1,SRqqt:1,FBt:1", PARTS)
    assert w == pytest.approx([1.0] * 6)


def test_ratios_are_preserved_and_mean_normalized():
    w = parse_particle_loss_weights("FRt:3,SRqqt:2,FBt:1", PARTS)
    assert sum(w) / len(w) == pytest.approx(1.0)      # loss scale preserved
    fr, sr, fb = w[0], w[2], w[4]
    assert fr / fb == pytest.approx(3.0)              # ratios intact
    assert sr / fb == pytest.approx(2.0)
    assert w[0] == pytest.approx(w[1])                # both FR targets equal
    assert w[2] == pytest.approx(w[3])
    assert w[4] == pytest.approx(w[5])


def test_two_to_one_ratio():
    w = parse_particle_loss_weights("FRt:2,SRqqt:1,FBt:1", PARTS)
    assert sum(w) / len(w) == pytest.approx(1.0)
    assert w[0] / w[2] == pytest.approx(2.0)
    assert w[2] == pytest.approx(w[4])


def test_unmatched_particles_default_to_one():
    w = parse_particle_loss_weights("FRt:2", PARTS)
    # SR and FB keep 1.0 pre-normalization -> ratio 2:1:1 after
    assert w[0] / w[2] == pytest.approx(2.0)
    assert w[2] == pytest.approx(w[4])
    assert sum(w) / len(w) == pytest.approx(1.0)


def test_longest_prefix_wins():
    # a specific particle overrides the topology-wide rule
    w = parse_particle_loss_weights("FRt:1,FRt2:5", PARTS)
    assert w[1] / w[0] == pytest.approx(5.0)


def test_typo_prefix_raises_rather_than_silently_doing_nothing():
    # a spec matching NOTHING is a typo, not a legitimate config
    with pytest.raises(ValueError, match="would do nothing"):
        parse_particle_loss_weights("FRtt:2", PARTS)


def test_malformed_entry_raises():
    with pytest.raises(ValueError, match="Malformed"):
        parse_particle_loss_weights("FRt=2", PARTS)


def test_resolved_only_event_file():
    # the SAME spec must be reusable on a resolved-only event file: the absent
    # topologies are ignored (not an error), leaving a uniform, mean-1 vector
    w = parse_particle_loss_weights("FRt:3,SRqqt:2,FBt:1", ["FRt1", "FRt2"])
    assert w == pytest.approx([1.0, 1.0])

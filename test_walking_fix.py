"""
Simulation test for the Walking fix (without a camera).

Builds a fake track history and checks:
  1. Standing (jitter)   -> should NOT become Walking
  2. Walking (drift)     -> should become Walking
  3. Stop after walking  -> should return to non-Walking
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pose_detection import _PersonTrack


def simulate(name, generator, frames=60):

    print(f"\n--- TEST: {name} ---")

    track = _PersonTrack(1, generator())

    results = []

    for i in range(frames):

        center = generator()

        track.update(center)

        is_walking, streak = track.update_walking_state()

        results.append(is_walking)

        if i % 10 == 0 or (i == frames - 1):
            state = "WALKING" if is_walking else "still"
            print(
                f"  frame {i:3d}: {state:8s} "
                f"(movement={track.average_movement():.1f}, "
                f"net={track.net_displacement():.1f}, "
                f"streak={streak})"
            )

    return results


# -----------------------------------------------------
# TEST 1: Standing — only jitter (random wobble)
# -----------------------------------------------------

random.seed(42)
jitter_count = {"n": 0}

def standing_jitter():
    jitter_count["n"] += 1
    return (100 + random.uniform(-4, 4),
            300 + random.uniform(-4, 4))

res1 = simulate(
    "Standing with jitter (Walking must NOT trigger)",
    standing_jitter
)

assert not any(res1), "FAIL: Standing jitter was classified as Walking!"
print("  PASS: Standing was never detected as Walking")


# -----------------------------------------------------
# TEST 2: Walking — consistent sideways drift + jitter
# -----------------------------------------------------

random.seed(42)
walk_x = {"x": 100.0}

def walking():
    walk_x["x"] += 5 + random.uniform(-1.5, 1.5)
    return (walk_x["x"], 300 + random.uniform(-3, 3))

res2 = simulate(
    "Walking (Walking SHOULD trigger)",
    walking
)

assert any(res2), "FAIL: Real walking was not detected at all!"
print(f"  PASS: Walking detected "
      f"({sum(res2)}/{len(res2)} frames)")


# -----------------------------------------------------
# TEST 3: Walking -> stop (back to standing)
# -----------------------------------------------------

random.seed(42)
mixed_x = {"x": 100.0, "phase": 0}

def walk_then_stop():
    if mixed_x["phase"] < 30:
        mixed_x["x"] += 5
    # Phase 2: stop, only jitter
    return (mixed_x["x"] + random.uniform(-3, 3),
            300 + random.uniform(-3, 3))

def gen():
    mixed_x["phase"] += 1
    return walk_then_stop()

res3 = simulate(
    "Walking -> Stop (Walking must turn OFF at the end)",
    gen,
    frames=70
)

assert any(res3[:40]), "FAIL: Walking was not detected at first!"
assert not any(res3[-15:]), "FAIL: Walking still shows after stopping!"
print("  PASS: Walking turned off after stopping")


print("\n" + "=" * 50)
print("ALL TESTS PASS! 🎉")
print("=" * 50)

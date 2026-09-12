"""
Walking fix ka simulation test (bina camera ke).

Fake track history bana kar check karta hai:
  1. Standing (jitter) -> Walking NAHI hona chahiye
  2. Walking (sideways drift) -> Walking hona chahiye
  3. Walking ke baad ruk jao -> Wapas non-Walking hona chahiye
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
# TEST 1: Standing — sirf jitter (random wobble)
# -----------------------------------------------------

random.seed(42)
jitter_count = {"n": 0}

def standing_jitter():
    jitter_count["n"] += 1
    return (100 + random.uniform(-4, 4),
            300 + random.uniform(-4, 4))

res1 = simulate(
    "Standing with jitter (Walking NAHI hona chahiye)",
    standing_jitter
)

assert not any(res1), "FAIL: Standing jitter ko Walking bata diya!"
print("  PASS: Standing kabhi Walking detect nahi hua")


# -----------------------------------------------------
# TEST 2: Walking — consistent sideways drift + jitter
# -----------------------------------------------------

random.seed(42)
walk_x = {"x": 100.0}

def walking():
    walk_x["x"] += 5 + random.uniform(-1.5, 1.5)
    return (walk_x["x"], 300 + random.uniform(-3, 3))

res2 = simulate(
    "Walking (Walking hona chahiye)",
    walking
)

assert any(res2), "FAIL: Asli walking detect hi nahi hua!"
print(f"  PASS: Walking detect hua "
      f"({sum(res2)}/{len(res2)} frames)")


# -----------------------------------------------------
# TEST 3: Walking -> ruk jao (wapas standing)
# -----------------------------------------------------

random.seed(42)
mixed_x = {"x": 100.0, "phase": 0}

def walk_then_stop():
    if mixed_x["phase"] < 30:
        mixed_x["x"] += 5
    # Phase 2: ruk jao, sirf jitter
    return (mixed_x["x"] + random.uniform(-3, 3),
            300 + random.uniform(-3, 3))

def gen():
    mixed_x["phase"] += 1
    return walk_then_stop()

res3 = simulate(
    "Walking -> Stop (end me Walking OFF hona chahiye)",
    gen,
    frames=70
)

assert any(res3[:40]), "FAIL: Pehle walking detect hi nahi hua!"
assert not any(res3[-15:]), "FAIL: Rukne ke baad bhi Walking dikh raha hai!"
print("  PASS: Rukne ke baad Walking band ho gaya")


print("\n" + "=" * 50)
print("SABHI TESTS PASS! 🎉")
print("=" * 50)

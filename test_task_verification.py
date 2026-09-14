"""
Task Verification ka simulation test (bina camera ke).

Synthetic keypoints bana kar check karta hai (test_walking_fix.py jaisa):
  1. "Raise your right hand" sirf tabhi COMPLETE hota hai jab right
     wrist sach me shoulder ke upar ho (geometry check).
  2. Temporal confirmation — 1-2 sahi frames se task COMPLETE nahi hota,
     TASK_CONFIRM_FRAMES consecutive frames chahiye.
  3. Person frame se gayab -> streak hold hota hai (false-positive safe).
  4. Wrong pose (left hand raised, right task) -> COMPLETE nahi hota.
  5. Multi-step sequential enforcement — Step 2/3 LOCKED rehte hain,
     sirf current step verify hota hai.
  6. Reset behavior — sab steps NOT COMPLETED, Step 1 IN PROGRESS.

Run:  python test_task_verification.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_detection import (
    TaskVerificationSession,
    evaluate_action,
    parse_task_text,
    detect_task,
    TASK_CONFIRM_FRAMES,
)


# -----------------------------------------------------
# Synthetic keypoint builders (CCO 17-keypoint layout)
# -----------------------------------------------------

def make_person(keypoints, person_conf=90.0, activity="Standing",
                confidence=90.0, justification=None):
    return {
        "keypoints": keypoints,
        "person_confidence": person_conf,
        "activity": activity,
        "confidence": confidence,
        "justification": justification or [],
    }


def standing_keypoints():
    """Seedha khada person (px coordinates, y neeche badhta hai)."""
    return {
        "nose": (320, 80),
        "left_shoulder": (270, 150), "right_shoulder": (370, 150),
        "left_elbow": (250, 230), "right_elbow": (390, 230),
        "left_wrist": (240, 300), "right_wrist": (400, 300),
        "left_hip": (285, 300), "right_hip": (355, 300),
        "left_knee": (280, 400), "right_knee": (360, 400),
        "left_ankle": (280, 500), "right_ankle": (360, 500),
    }


def sitting_keypoints():
    """Chair par baitha person — hips knee-level par, knees bent."""
    return {
        "nose": (320, 150),
        "left_shoulder": (270, 220), "right_shoulder": (370, 220),
        "left_elbow": (250, 300), "right_elbow": (390, 300),
        "left_wrist": (250, 380), "right_wrist": (390, 380),
        "left_hip": (280, 380), "right_hip": (360, 380),
        "left_knee": (400, 390), "right_knee": (400, 400),
        "left_ankle": (400, 490), "right_ankle": (400, 500),
    }


def right_hand_raised(keep_hips_standing=True):
    """Standing person jiska right wrist shoulder ke upar hai."""
    kp = standing_keypoints() if keep_hips_standing else sitting_keypoints()
    kp["right_elbow"] = (410, 80)
    kp["right_wrist"] = (420, 20)     # shoulder (150) se ~130px upar
    return kp


def left_hand_raised():
    kp = standing_keypoints()
    kp["left_elbow"] = (230, 80)
    kp["left_wrist"] = (220, 20)
    return kp


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


def feed_frames(session, persons, frames):
    """N frames tak same person list feed karo, results wapas do."""
    results = []
    for _ in range(frames):
        results.append(session.process_persons(persons))
    return results


# -----------------------------------------------------
print("\n--- TEST 1: Task parsing ---")
check("'Raise your right hand' -> raise_right_hand",
      parse_task_text("Raise your right hand") == "raise_right_hand")
check("'Sit down' -> sit", parse_task_text("Sit down") == "sit")
check("'Stand up' -> stand", parse_task_text("Stand up") == "stand")
check("'Walk' -> walk", parse_task_text("Walk") == "walk")
check("'Raise both hands' -> raise_both_hands",
      parse_task_text("Raise both hands") == "raise_both_hands")
check("Garbage -> None", parse_task_text("Do a backflip") is None)

# -----------------------------------------------------
print("\n--- TEST 2: Single-frame geometry detection ---")

detected, conf, reasons = evaluate_action(
    "raise_right_hand", [make_person(right_hand_raised())]
)
check("Right hand raised -> detected", detected)
check("Confidence in realistic range (50-98)", 50 <= conf <= 98,
      f"(got {conf})")

detected, _, _ = evaluate_action(
    "raise_right_hand", [make_person(standing_keypoints())]
)
check("Hands down -> NOT detected", not detected)

detected, _, _ = evaluate_action(
    "raise_right_hand", [make_person(left_hand_raised())]
)
check("LEFT hand raised -> right-hand task NOT detected", not detected)

detected, _, _ = evaluate_action(
    "raise_right_hand", [make_person(right_hand_raised(keep_hips_standing=False))]
)
check("Right hand raised while sitting -> detected", detected)

detected, _, _ = evaluate_action("sit", [make_person(sitting_keypoints())])
check("Sitting pose -> sit detected", detected)

detected, _, _ = evaluate_action("sit", [make_person(standing_keypoints())])
check("Standing pose -> sit NOT detected", not detected)

detected, _, _ = evaluate_action(
    "stand", [make_person(standing_keypoints())]
)
check("Standing pose -> stand detected", detected)

detected, _, _ = evaluate_action(
    "raise_both_hands", [make_person(standing_keypoints())]
)
check("Hands down -> both-hands NOT detected", not detected)

both_up = standing_keypoints()
both_up["right_elbow"] = (410, 80)
both_up["right_wrist"] = (420, 20)
both_up["left_elbow"] = (230, 80)
both_up["left_wrist"] = (220, 20)
detected, _, _ = evaluate_action(
    "raise_both_hands", [make_person(both_up)]
)
check("Both hands up -> detected", detected)

no_person = detect_task("Walk", [])
check("No person -> safely not detected", not no_person["detected"])

missing_kp = make_person({"left_shoulder": (0, 0), "right_shoulder": (0, 0)})
detected, _, _ = evaluate_action("raise_right_hand", [missing_kp])
check("Missing/zero keypoints -> safely not detected", not detected)

# -----------------------------------------------------
print("\n--- TEST 3: Single task temporal confirmation ---")

session = TaskVerificationSession(["Raise your right hand"])
check("Initial state IN PROGRESS",
      session.steps[0]["state"] == "IN PROGRESS")

# Ek frame detected — abhi COMPLETE nahi hona chahiye
r = session.process_persons([make_person(right_hand_raised())])
check("1 good frame -> still IN PROGRESS (no false positive)",
      not r["step_completed"] and r["streak"] == 1)

# Kam frames (aadhe) — abhi bhi nahi
feed_frames(session, [make_person(right_hand_raised())],
            TASK_CONFIRM_FRAMES // 2 - 1)
check(f"{TASK_CONFIRM_FRAMES // 2} good frames -> still not completed",
      not session.steps[0]["state"] == "COMPLETED")

# Galat pose 1 frame — streak decay hota hai par complete nahi
session.process_persons(make_person(standing_keypoints()))
check("Wrong-pose frame does not complete the step",
      session.steps[0]["state"] != "COMPLETED")

# Ab lagatar sahi frames — sliding window me 12 detections
# poore hote hi complete ho jaana chahiye (flicker ke baad bhi)
completion_result = None
for i in range(TASK_CONFIRM_FRAMES + 2):
    r = session.process_persons([make_person(right_hand_raised())])
    if r["step_completed"]:
        completion_result = r
        break

check(f"{TASK_CONFIRM_FRAMES} detections in window -> COMPLETED",
      session.steps[0]["state"] == "COMPLETED")
check("Step completion reported",
      completion_result is not None
      and completion_result["step_completed"])
check("All completed (single task)",
      completion_result is not None
      and completion_result["all_completed"])
check("Confidence recorded from real geometry",
      session.steps[0]["confidence"] >= 50,
      f"(got {session.steps[0]['confidence']})")
check("History event generated",
      completion_result is not None
      and any(e["status"] == "Completed"
              for e in completion_result["events"]))

# -----------------------------------------------------
print("\n--- TEST 4: Person missing holds streak (no false positive) ---")

session2 = TaskVerificationSession(["Stand"])
for i in range(TASK_CONFIRM_FRAMES - 2):
    session2.process_persons(make_person(standing_keypoints()))

# 3 frame ke liye person gayab
for i in range(3):
    r = session2.process_persons([])
check("Person missing -> streak held (not reset, not completed)",
      0 < r["streak"] < TASK_CONFIRM_FRAMES,
      f"(streak={r['streak']})")

# Wapas aakar confirm karo
for i in range(3):
    r = session2.process_persons(make_person(standing_keypoints()))
check("Streak resumes -> task completes after full confirmation",
      r["all_completed"], f"(streak={r['streak']})")

# -----------------------------------------------------
print("\n--- TEST 5: Multi-step sequential enforcement ---")

session3 = TaskVerificationSession(
    ["Stand up", "Raise your right hand", "Sit down"]
)

states = [(s["text"], s["state"]) for s in session3.steps_view()]
check("Initial: Step 1 IN PROGRESS", states[0][1] == "IN PROGRESS")
check("Initial: Step 2 LOCKED", states[1][1] == "LOCKED")
check("Initial: Step 3 LOCKED", states[2][1] == "LOCKED")

# Step 2 ka pose do (RIGHT HAND UP par baitha hua — taaki wo
# Step 1 'Stand up' se match NA kare) — LOCKED hai, kuch nahi hoga
feed_frames(session3,
            [make_person(right_hand_raised(keep_hips_standing=False))],
            TASK_CONFIRM_FRAMES + 5)
check("Future step pose ignored while Step 1 active",
      session3.current_index == 0)

# Step 1 (Stand) complete karo
for i in range(TASK_CONFIRM_FRAMES):
    r = session3.process_persons(make_person(standing_keypoints()))

check("Step 1 completed after temporal confirmation",
      session3.steps[0]["state"] == "COMPLETED")
check("Step 2 now IN PROGRESS",
      session3.steps[1]["state"] == "IN PROGRESS")
check("Step 3 still LOCKED",
      session3.steps[2]["state"] == "LOCKED")
check("Not all completed yet", not r["all_completed"])

# Step 3 ka pose (sitting) do — Step 2 abhi pending hai
feed_frames(session3, [make_person(sitting_keypoints())],
            TASK_CONFIRM_FRAMES + 5)
check("Step 3 pose ignored while Step 2 active",
      session3.steps[1]["state"] == "IN PROGRESS"
      and session3.steps[2]["state"] == "LOCKED")

# Step 2 (right hand) complete karo
for i in range(TASK_CONFIRM_FRAMES):
    r = session3.process_persons(make_person(right_hand_raised()))

check("Step 2 completed",
      session3.steps[1]["state"] == "COMPLETED")
check("Step 3 now IN PROGRESS",
      session3.steps[2]["state"] == "IN PROGRESS")
check("Step 3 NOT COMPLETED before its turn is done",
      session3.steps[2]["state"] != "COMPLETED")

# Step 3 (Sit) complete karo
for i in range(TASK_CONFIRM_FRAMES):
    r = session3.process_persons(make_person(sitting_keypoints()))

check("Step 3 completed", session3.steps[2]["state"] == "COMPLETED")
check("ALL TASKS COMPLETED", r["all_completed"])
check("Final event recorded",
      any(e["status"] == "All Tasks Completed" for e in r["events"]))

# -----------------------------------------------------
print("\n--- TEST 6: Reset behavior ---")

session3.reset()
check("After reset: Step 1 IN PROGRESS",
      session3.steps[0]["state"] == "IN PROGRESS")
check("After reset: Step 2 NOT COMPLETED (unlocked)",
      session3.steps[1]["state"] == "NOT COMPLETED")
check("After reset: Step 3 NOT COMPLETED (unlocked)",
      session3.steps[2]["state"] == "NOT COMPLETED")
check("After reset: all_completed cleared",
      not session3.all_completed and session3.current_index == 0)

# -----------------------------------------------------
print("\n--- TEST 7: Unknown task text handled safely ---")

session4 = TaskVerificationSession(["Do a cartwheel"])
r = session4.process_persons([make_person(standing_keypoints())])
check("Unknown task -> active False, no crash", not r["active"])
check("Unknown task -> honest reason given",
      any("not recognized" in x for x in r["reasons"]))

# -----------------------------------------------------
print(f"\n{'=' * 50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'=' * 50}")

sys.exit(1 if failed else 0)

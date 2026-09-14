"""
step_validator.py
=================

Sequential Step Evaluation for BAS•AI.

Validates a fixed activity sequence (Standing -> Walking -> Sitting)
against the live camera feed:

  - Correct step detected  -> status "CORRECT"   (sequence advances)
  - Wrong step detected    -> status "WRONG"     (camera.py gives a voice alert)
  - All steps complete     -> status "COMPLETED", get_current_step() -> None
  - Pose unclear (Unknown) -> status "NEUTRAL"   (UI shows "WAITING")
"""

# ------------------------------------------------------------
# SEQUENCE
# ------------------------------------------------------------

STEP_SEQUENCE = ["Standing", "Walking", "Sitting"]


class StepValidator:
    """Sequential activity validator (stored in Streamlit session_state)."""

    def __init__(self, sequence=None):
        self.sequence = list(sequence) if sequence else list(STEP_SEQUENCE)
        self.current_index = 0

    # --------------------------------------------------------
    # CORE API (used by camera.py)
    # --------------------------------------------------------

    def check_activity(self, activity):
        """
        Compare the detected activity with the current step of the sequence.

        Returns:
            {"status": "CORRECT" | "WRONG" | "COMPLETED" | "NEUTRAL",
             "message": str}
        """
        # All steps have been completed
        if self.current_index >= len(self.sequence):
            return {
                "status": "COMPLETED",
                "message": "All steps completed. 🎉",
            }

        # Pose unclear (legs not visible) — a verdict would be wrong
        if not activity or activity == "Unknown":
            return {
                "status": "NEUTRAL",
                "message": "Pose unclear — step could not be verified.",
            }

        expected = self.sequence[self.current_index]

        if activity == expected:
            self.current_index += 1

            if self.current_index >= len(self.sequence):
                return {
                    "status": "COMPLETED",
                    "message": f"Final step '{expected}' complete. 🎉",
                }

            next_step = self.sequence[self.current_index]
            return {
                "status": "CORRECT",
                "message": f"Step complete: {expected}. Now: {next_step}",
            }

        return {
            "status": "WRONG",
            "message": f"Wrong step. Expected {expected}, but detected {activity}.",
        }

    def get_current_step(self):
        """Name of the currently expected step, or None if the sequence is complete."""
        if self.current_index >= len(self.sequence):
            return None
        return self.sequence[self.current_index]

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def reset(self):
        """Restart the sequence from the beginning."""
        self.current_index = 0

    def get_progress(self):
        """(completed_steps, total_steps) — for UI progress."""
        return min(self.current_index, len(self.sequence)), len(self.sequence)

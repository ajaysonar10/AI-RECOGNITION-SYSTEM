"""
step_validator.py
=================

Sequential Step Evaluation for BAS•AI.

Fixed activity sequence (Standing -> Walking -> Sitting) ko live camera
feed ke against validate karta hai:

  - Sahi step detect hua   -> status "CORRECT"   (sequence aage badhti hai)
  - Galat step detect hua  -> status "WRONG"     (camera.py voice alert deta hai)
  - Saare steps complete   -> status "COMPLETED", get_current_step() -> None
  - Pose unclear (Unknown) -> status "NEUTRAL"   (UI "WAITING" dikhata hai)
"""

# ------------------------------------------------------------
# SEQUENCE
# ------------------------------------------------------------

STEP_SEQUENCE = ["Standing", "Walking", "Sitting"]


class StepValidator:
    """Sequential activity validator (Streamlit session_state me store hota hai)."""

    def __init__(self, sequence=None):
        self.sequence = list(sequence) if sequence else list(STEP_SEQUENCE)
        self.current_index = 0

    # --------------------------------------------------------
    # CORE API (camera.py isko use karta hai)
    # --------------------------------------------------------

    def check_activity(self, activity):
        """
        Detected activity ko sequence ke current step se compare karo.

        Returns:
            {"status": "CORRECT" | "WRONG" | "COMPLETED" | "NEUTRAL",
             "message": str}
        """
        # Saare steps complete ho chuke hain
        if self.current_index >= len(self.sequence):
            return {
                "status": "COMPLETED",
                "message": "All steps completed. 🎉",
            }

        # Pose unclear (legs visible nahi) — verdict dena galat hoga
        if not activity or activity == "Unknown":
            return {
                "status": "NEUTRAL",
                "message": "Pose unclear — step verify nahi ho paya.",
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
        """Abhi expected step ka naam, ya None agar sequence complete hai."""
        if self.current_index >= len(self.sequence):
            return None
        return self.sequence[self.current_index]

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def reset(self):
        """Sequence dobara shuru karne ke liye."""
        self.current_index = 0

    def get_progress(self):
        """(completed_steps, total_steps) — UI progress ke liye."""
        return min(self.current_index, len(self.sequence)), len(self.sequence)

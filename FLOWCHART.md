# 🗺️ BAS-AI Program Flowcharts

Visual flow of the AI-RECOGNITION-SYSTEM. These diagrams render directly on GitHub (Mermaid).

> 🖼️ **PNG exports for presentations/reports:** [`diagrams/`](diagrams/) contains all 5 diagrams as high-resolution PNGs (2× scale):
> [`01_system_pipeline.png`](diagrams/01_system_pipeline.png) · [`02_activity_classification.png`](diagrams/02_activity_classification.png) · [`03_task_state_machine.png`](diagrams/03_task_state_machine.png) · [`04_object_hold_detection.png`](diagrams/04_object_hold_detection.png) · [`05_desktop_launch.png`](diagrams/05_desktop_launch.png)
> Edit the `.mmd` sources in the same folder to change a diagram, then re-export with mermaid-cli.

---

## 1. Overall System Pipeline (every frame)

![Overall System Pipeline](diagrams/01_system_pipeline.png)

```mermaid
flowchart TD
    A["📷 Camera Source<br/>(Laptop webcam or Mobile IP cam)"] --> B["cv2.VideoCapture<br/>frame read"]
    B --> C["🧍 pose_detection.analyze_frame()<br/>YOLO11n-pose → 17 keypoints"]
    B --> D["📦 object_detection.detect_objects()<br/>YOLO11n → 80 COCO classes"]
    C --> E["Person Tracker<br/>nearest-center matching → Track ID"]
    E --> F["Activity + Justification<br/>(Standing / Sitting / Walking / Unknown)"]
    D --> G["ObjectTracker<br/>persistent tracks + hold/lift state"]
    F --> H["Task Layer"]
    G --> H
    H --> I["StepValidator<br/>fixed sequence"]
    H --> J["TaskVerificationSession<br/>pose + object tasks"]
    H --> K["LiveTaskEvaluator<br/>free-text tasks"]
    I --> L["🖥️ Streamlit UI<br/>metrics · checklists · history · analytics"]
    J --> L
    K --> L
    F --> M["🔊 pyttsx3 Voice Alert<br/>(wrong activity)"]
    L --> N["BGR → RGB<br/>annotated frame display"]
```

---

## 2. Activity Classification Logic (`pose_detection.py`)

![Activity Classification Logic](diagrams/02_activity_classification.png)

```mermaid
flowchart TD
    S(["Person detected"]) --> K{"17 keypoints<br/>available?"}
    K -- "No" --> U["❓ Unknown (0%)"]
    K -- "Yes" --> W{"Walking test passes?<br/>movement > 4 px/frame<br/>AND net displacement > 25 px<br/>AND consistency ≥ 0.55<br/>AND 4-frame streak"}
    W -- "Yes" --> WK["🚶 Walking<br/>(conf 75–95%)"]
    W -- "No" --> KA{"Avg knee angle?"}
    KA -- "< 125°" --> SIT["🪑 Sitting<br/>(knees bent)"]
    KA -- "≥ 155°" --> ST["🧍 Standing<br/>(legs straight)"]
    KA -- "125°–155°" --> U2["❓ Unknown<br/>(pose not clear)"]
    WK --> R["Return activity<br/>+ confidence + reasons"]
    SIT --> R
    ST --> R
    U2 --> R
    U --> R
```

---

## 3. Task Verification State Machine (`task_detection.py`)

![Task Verification State Machine](diagrams/03_task_state_machine.png)

```mermaid
flowchart TD
    T(["Task text entered"]) --> P{"Parse text"}
    P -- "pose phrase" --> PD["Pose detector<br/>(stand / sit / raise hand / walk)"]
    P -- "object phrase" --> OD["Object handler<br/>(pick / place / handoff / 22 tasks)"]
    P -- "unknown / unsupported" --> REJ["🚫 Honest rejection<br/>never a fake result"]
    PD --> E{"Pose/object matched<br/>this frame?"}
    OD --> E
    E -- "Yes" --> INC["streak +1<br/>(window of 18 frames)"]
    E -- "No" --> HOLD["window unchanged<br/>(flicker-tolerant)"]
    INC --> Q{"streak ≥ 12<br/>confirmations?"}
    HOLD --> Q
    Q -- "No" --> LOOP["Continue camera loop"]
    Q -- "Yes" --> DONE["✅ Step COMPLETED<br/>event + timestamp + confidence"]
    DONE --> MORE{"More steps?"}
    MORE -- "Yes" --> NEXT["Next step unlocks<br/>(sequential enforcement)"] --> LOOP
    MORE -- "No" --> ALL["🎉 ALL TASKS COMPLETED<br/>→ task history"]
    REJ --> UI
    LOOP --> UI["🖥️ Live UI status<br/>streak · confidence · reasons"]
```

---

## 4. Object Hold Detection (`object_tracking.py`)

![Object Hold Detection](diagrams/04_object_hold_detection.png)

```mermaid
flowchart TD
    O(["Object track alive"]) --> P{"Wrist inside reach?<br/>max(0.7 × torso, 0.5 × box diag)"}
    P -- "Yes, 6 frames" --> H["✋ HELD<br/>lift baseline captured"]
    P -- "No" --> F["Free object"]
    H --> S{"Wrist OR elbow<br/>still near?"}
    S -- "Yes" --> L["Track lift<br/>max_lift updated"]
    S -- "No, 2 frames" --> REL["Released<br/>(baseline RETAINED<br/>for re-grip)"]
    F --> REST{"Free + stable +<br/>no hand 8 frames?"}
    REST -- "Yes" --> RB["Reset baseline<br/>(object at rest)"]
    REST -- "No" --> F
    L --> S
    REL --> P
```

---

## 5. Desktop App Launch Flow (`desktop_app.py`)

![Desktop App Launch Flow](diagrams/05_desktop_launch.png)

```mermaid
flowchart TD
    M(["BAS-AI.exe / desktop_app.py"]) --> FP["Find free port<br/>(prefer 8765)"]
    FP --> SV["Start Streamlit server<br/>(headless, localhost)"]
    SV --> HL{"Health check<br/>within 90 s?"}
    HL -- "Yes" --> WV{"WebView2<br/>available?"}
    WV -- "Yes" --> NW["🖥️ Native window<br/>(pywebview 1500×920)"]
    WV -- "No" --> BR["🌐 Default browser fallback"]
    HL -- "No" --> BR
    NW --> CL["Window closed →<br/>stop server"]
    BR --> CL
```

---

*Companion doc: `PROGRAM_WORKING_STEP_BY_STEP.txt` (full text analysis)*

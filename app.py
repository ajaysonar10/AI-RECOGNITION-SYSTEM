import streamlit as st
from datetime import datetime
from camera import show_camera, show_task_evaluation, show_task_verification

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BAS-AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN APP ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 10%,
                rgba(30, 100, 180, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 10% 90%,
                rgba(70, 40, 140, 0.15),
                transparent 30%
            ),
            #030812;
        color: white;
    }

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 25px;
        padding-bottom: 40px;
        max-width: 1450px;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #050d18;
        border-right: 1px solid #182d47;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 25px 20px;
    }


    /* ---------- HEADINGS ---------- */

    h1,
    h2,
    h3 {
        color: #ffffff !important;
    }


    /* ---------- METRIC CARDS ---------- */

    div[data-testid="metric-container"] {
        background: #081522;
        border: 1px solid #19334f;
        border-radius: 14px;
        padding: 18px;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        background: #1689d8;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 10px 22px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #25a4f5;
        color: white;
    }


    /* ---------- DATAFRAME ---------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid #19334f;
        border-radius: 12px;
        overflow: hidden;
    }


    /* ---------- INPUT FIELDS ---------- */

    .stTextArea textarea,
    .stTextInput input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: #081522;
        border: 1px solid #19334f;
        border-radius: 10px;
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #526981 !important;
        -webkit-text-fill-color: #526981 !important;
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #1689d8 !important;
        outline: none;
    }


    /* ---------- INFO BOX ---------- */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* ---------- FOOTER ---------- */

    .footer-text {
        text-align: center;
        color: #526981;
        font-size: 11px;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# REAL AI ACTIVITY DATA
# ============================================================

if "activity_history" not in st.session_state:
    st.session_state.activity_history = []

# Task verification history (activity history se logically alag)
if "task_history" not in st.session_state:
    st.session_state.task_history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # 🛰️ BAS•AI

        **ON-BOARD SPACE INTELLIGENCE**

        ---
        """,
    )

    st.caption("MAIN NAVIGATION")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📡 Live Monitor",
            "🕒 Activity History",
            "📊 Analytics"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 🛰️ Mission")

    st.caption("BAS EXPERIMENT")

    st.success("SYSTEM ONLINE")

    st.caption("SIH 2026")


# ============================================================
# TOP HEADER
# ============================================================

header_col1, header_col2, header_col3 = st.columns(
    [2, 5, 1]
)

with header_col1:

    st.markdown(
        "## 🛰️ BAS<span style='color:#55baff'>•</span>AI",
        unsafe_allow_html=True
    )

with header_col2:

    st.markdown(
        "<p style='text-align:center; "
        "color:#7187a3; "
        "font-size:12px; "
        "letter-spacing:2px;'>"
        "HUMAN ACTIVITY RECOGNITION SYSTEM"
        "</p>",
        unsafe_allow_html=True
    )

with header_col3:

    st.success("ONLINE")


st.divider()


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    # --------------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------------

    st.markdown("### ON-BOARD AI • BAS EXPERIMENT • SIH 2026")

    st.title(
        "Understanding Human Activity in Space 🚀"
    )

    st.write(
        "BAS-AI is an intelligent on-board AI system "
        "designed to recognize, classify and monitor "
        "human activities during space-based experiments."
    )

    st.info(
        "🛰️ MISSION ACTIVE   |   AI ENGINE READY"
    )

    st.divider()


    # --------------------------------------------------------
    # CORE INTELLIGENCE
    # --------------------------------------------------------

    st.subheader("Core Intelligence")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🤖")

        st.markdown(
            "**Human Activity Recognition**"
        )

        st.caption(
            "AI identifies and classifies human "
            "activities from experiment data."
        )


    with col2:

        st.markdown("### 👨‍🚀")

        st.markdown(
            "**Crew Monitoring**"
        )

        st.caption(
            "Monitor crew activity patterns and "
            "maintain continuous mission awareness."
        )


    with col3:

        st.markdown("### ⚡")

        st.markdown(
            "**Anomaly Detection**"
        )

        st.caption(
            "Identify unusual activity patterns "
            "and highlight important events."
        )


    st.divider()


    # --------------------------------------------------------
    # MISSION INFORMATION
    # --------------------------------------------------------

    st.subheader("Mission Information")

    info1, info2, info3 = st.columns(3)

    with info1:

        st.markdown("**🛰️ Experiment**")

        st.caption(
            "BAS On-Board Experiment"
        )

    with info2:

        st.markdown("**🧠 AI Module**")

        st.caption(
            "Human Activity Recognition"
        )

    with info3:

        st.markdown("**🏆 Event**")

        st.caption(
            "Smart India Hackathon 2026"
        )


# ============================================================
# LIVE MONITOR
# ============================================================

elif page == "📡 Live Monitor":

    st.title("📡 Live Monitor")

    st.caption(
        "Real-time monitoring + live task evaluation"
    )

    # Live Monitor now shows only Task Verification
    # (Activity Monitor aur Task Evaluation code camera.py me
    #  intact hai — bas navigation se hataya gaya hai)
    show_task_verification()

    
    


# ============================================================
# ACTIVITY HISTORY
# ============================================================

elif page == "🕒 Activity History":

    st.title("🕒 Activity History")

    st.caption(
        "Previously detected human activities"
    )

    st.divider()

    # Real detections collected by the AI camera
    history = st.session_state.get(
        "activity_history",
        []
    )

    if history:

        history_rows = []

        for item in history:
            history_rows.append(
                {
                    "Date": item["date"],
                    "Time": item["time"],
                    "Activity": item["activity"],
                    "Confidence": f'{item["confidence"]:.1f}%',
                    "Status": item["status"]
                }
            )

        st.dataframe(
            history_rows,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Activity Summary")

        total = len(history)
        normal = sum(
            1 for item in history
            if item["status"] == "Normal"
        )
        anomalies = total - normal

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Activities",
                total
            )

        with col2:
            st.metric(
                "Normal Activities",
                normal
            )

        with col3:
            st.metric(
                "Anomalies",
                anomalies
            )

    else:

        st.info(
            "No AI activity detections yet. "
            "Start the camera from Live Monitor."
        )

    # --------------------------------------------------------
    # TASK VERIFICATION HISTORY (activity history se alag)
    # --------------------------------------------------------

    task_history = st.session_state.get("task_history", [])

    if task_history:

        st.divider()

        st.subheader("✅ Task Verification History")

        st.caption(
            "Tasks completed via Task Verification "
            "(kept separate from activity history)"
        )

        task_rows = []

        for item in task_history:
            task_rows.append(
                {
                    "Date": item["date"],
                    "Time": item["time"],
                    "Task": item["task"],
                    "Step": item["step"],
                    "Status": item["status"],
                    "Confidence": f'{item["confidence"]:.0f}%'
                }
            )

        st.dataframe(
            task_rows,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics")

    st.caption(
        "Human activity recognition analytics"
    )

    st.divider()

    history = st.session_state.get(
        "activity_history",
        []
    )

    if history:

        total = len(history)

        average_confidence = (
            sum(
                item["confidence"]
                for item in history
            ) / total
        )

        anomalies = sum(
            1
            for item in history
            if item["status"] != "Normal"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Activities",
                total
            )

        with col2:
            st.metric(
                "Average Confidence",
                f"{average_confidence:.1f}%"
            )

        with col3:
            st.metric(
                "Anomalies",
                anomalies
            )

        st.subheader("Activity Distribution")

        activity_counts = {}

        for item in history:
            activity = item["activity"]
            activity_counts[activity] = (
                activity_counts.get(activity, 0) + 1
            )

        st.bar_chart(activity_counts)

        st.subheader("AI Model Performance")

        confidence_value = max(
            0.0,
            min(1.0, average_confidence / 100)
        )

        st.progress(confidence_value)

        st.caption(
            f"Average model confidence: "
            f"{average_confidence:.1f}%"
        )

    else:

        st.info(
            "No AI data available yet. "
            "Start the camera from Live Monitor "
            "to generate analytics."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        BAS•AI | ON-BOARD HUMAN ACTIVITY RECOGNITION
        <br><br>
        SMART INDIA HACKATHON 2026
    </div>
    """,
    unsafe_allow_html=True
)
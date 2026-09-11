import streamlit as st
from datetime import datetime

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

    st.caption("SYSTEM")

    system_page = st.radio(
        "System Navigation",
        [
            "⚙️ Settings",
            "ℹ️ About",
            "📋 Logs",
            "🟢 System Status"
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
    # MISSION OVERVIEW
    # --------------------------------------------------------

    st.subheader("Mission Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            label="System Status",
            value="ONLINE"
        )

    with col2:

        st.metric(
            label="AI Confidence",
            value="96.8%"
        )

    with col3:

        st.metric(
            label="Activities Detected",
            value="08"
        )

    with col4:

        st.metric(
            label="Mission Time",
            value="142:36"
        )


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
    # RECENT ACTIVITY
    # --------------------------------------------------------

    st.subheader("Recent Activity")

    activity_col1, activity_col2 = st.columns(
        [1.5, 1]
    )


    with activity_col1:

        activity_data = {
            "Time": [
                "21:26:32",
                "21:26:27",
                "21:26:21",
                "21:25:58"
            ],
            "Activity": [
                "Walking",
                "Standing",
                "Walking",
                "Sitting"
            ],
            "Confidence": [
                "98.4%",
                "96.7%",
                "97.9%",
                "94.2%"
            ]
        }

        st.dataframe(
            activity_data,
            use_container_width=True,
            hide_index=True
        )


    with activity_col2:

        st.markdown("### 🧠 AI Inference Engine")

        st.write(
            "The AI engine is actively processing "
            "human activity observations."
        )

        st.progress(0.96)

        st.metric(
            "Model Confidence",
            "96%"
        )


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
        "Real-time human activity monitoring"
    )

    st.divider()


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current Activity",
            "Walking"
        )

    with col2:

        st.metric(
            "Confidence",
            "98.4%"
        )

    with col3:

        st.metric(
            "System",
            "ONLINE"
        )


    st.subheader("Current AI Detection")

    st.progress(0.984)

    st.success(
        "🟢 Walking activity detected"
    )


    st.subheader("Live Activity Stream")

    st.write(
        "🟢 21:26:32  —  Walking  —  98.4%"
    )

    st.write(
        "🟢 21:26:27  —  Standing  —  96.7%"
    )

    st.write(
        "🟢 21:26:21  —  Walking  —  97.9%"
    )

    st.write(
        "🟢 21:25:58  —  Sitting  —  94.2%"
    )


# ============================================================
# ACTIVITY HISTORY
# ============================================================

elif page == "🕒 Activity History":

    st.title("🕒 Activity History")

    st.caption(
        "Previously detected human activities"
    )

    st.divider()


    history_data = {
        "Date": [
            "11 Sep 2026",
            "11 Sep 2026",
            "11 Sep 2026",
            "11 Sep 2026",
            "11 Sep 2026"
        ],

        "Time": [
            "21:26:32",
            "21:26:27",
            "21:26:21",
            "21:25:58",
            "21:25:42"
        ],

        "Activity": [
            "Walking",
            "Standing",
            "Walking",
            "Sitting",
            "Working"
        ],

        "Confidence": [
            "98.4%",
            "96.7%",
            "97.9%",
            "94.2%",
            "91.8%"
        ],

        "Status": [
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal"
        ]
    }


    st.dataframe(
        history_data,
        use_container_width=True,
        hide_index=True
    )


    st.divider()

    st.subheader("Activity Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Activities",
            "1,284"
        )

    with col2:

        st.metric(
            "Normal Activities",
            "1,281"
        )

    with col3:

        st.metric(
            "Anomalies",
            "03"
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


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Activities",
            "1,284"
        )

    with col2:

        st.metric(
            "Average Confidence",
            "95.7%"
        )

    with col3:

        st.metric(
            "Anomalies",
            "03"
        )


    st.subheader("Activity Distribution")

    chart_data = {
        "Walking": 35,
        "Standing": 28,
        "Sitting": 20,
        "Working": 17
    }

    st.bar_chart(chart_data)


    st.subheader("AI Model Performance")

    st.progress(0.957)

    st.caption(
        "Average model confidence: 95.7%"
    )


# ============================================================
# SETTINGS
# ============================================================

elif system_page == "⚙️ Settings":

    st.title("⚙️ Settings")

    st.caption(
        "Configure BAS-AI system parameters"
    )

    st.divider()


    st.subheader("AI Configuration")

    confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.50,
        max_value=1.00,
        value=0.80,
        step=0.01
    )

    st.write(
        f"Current threshold: {confidence:.2f}"
    )


    realtime = st.toggle(
        "Real-time monitoring",
        value=True
    )

    anomaly = st.toggle(
        "Anomaly detection",
        value=True
    )


    st.subheader("Interface")

    interface_mode = st.selectbox(
        "Interface Mode",
        [
            "Dark Space",
            "Standard"
        ]
    )


    if st.button("Save Settings"):

        st.success(
            "Settings saved successfully."
        )


# ============================================================
# ABOUT
# ============================================================

elif system_page == "ℹ️ About":

    st.title("ℹ️ About BAS-AI")

    st.caption(
        "On-Board Human Activity Recognition System"
    )

    st.divider()


    st.subheader("What is BAS-AI?")

    st.write(
        "BAS-AI is an AI-based human activity "
        "recognition system designed to monitor "
        "and analyze human activities during "
        "on-board space experiments."
    )


    st.subheader("Core Functions")

    st.write(
        "🤖 Human Activity Recognition"
    )

    st.write(
        "👨‍🚀 Crew Activity Monitoring"
    )

    st.write(
        "📊 Activity Analytics"
    )

    st.write(
        "⚡ Anomaly Detection"
    )

    st.write(
        "🕒 Activity History"
    )


    st.divider()

    st.info(
        "Smart India Hackathon 2026"
    )


# ============================================================
# LOGS
# ============================================================

elif system_page == "📋 Logs":

    st.title("📋 System Logs")

    st.caption(
        "BAS-AI system activity logs"
    )

    st.divider()


    st.code(
        """
21:26:32  [INFO] Activity detected: WALKING
21:26:32  [INFO] Confidence: 98.4%

21:26:27  [INFO] Activity detected: STANDING
21:26:27  [INFO] Confidence: 96.7%

21:26:21  [INFO] AI inference completed
21:26:21  [INFO] Processing time: 42 ms

21:25:58  [INFO] Activity detected: SITTING
21:25:58  [INFO] Confidence: 94.2%

21:25:42  [SYSTEM] Model loaded successfully
21:25:40  [SYSTEM] BAS-AI started
        """,
        language="text"
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

elif system_page == "🟢 System Status":

    st.title("🟢 System Status")

    st.caption(
        "BAS-AI component health monitoring"
    )

    st.divider()


    col1, col2 = st.columns(2)


    with col1:

        st.subheader("System Components")

        st.success(
            "🟢 AI Engine — ONLINE"
        )

        st.success(
            "🟢 Activity Recognition — ONLINE"
        )

        st.success(
            "🟢 Data Pipeline — ONLINE"
        )

        st.success(
            "🟢 Monitoring — ACTIVE"
        )


    with col2:

        st.subheader("Performance")

        st.metric(
            "CPU Usage",
            "24%"
        )

        st.metric(
            "Memory Usage",
            "41%"
        )

        st.metric(
            "AI Latency",
            "42 ms"
        )


    st.divider()

    st.info(
        "All major BAS-AI components are operating normally."
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
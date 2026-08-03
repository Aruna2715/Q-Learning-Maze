import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="MazeMind | Q-Learning Maze Solver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM LIGHT THEME
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(218, 226, 255, 0.78),
                transparent 32%
            ),
            radial-gradient(
                circle at bottom right,
                rgba(203, 245, 229, 0.72),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #fbfcff 0%,
                #f7f5ff 50%,
                #f2fbf8 100%
            );
        color: #1d2a3a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #ffffff 0%,
                #f2f4ff 48%,
                #edf9f5 100%
            );
        border-right: 1px solid #d9dfeb;
        box-shadow: 6px 0 24px rgba(44, 62, 92, 0.08);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.3rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #17243a !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px;
        display: flex;
        flex-direction: column;
        margin-top: 10px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #d8deeb;
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 2px;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #edf0ff;
        border-color: #8b91e8;
        transform: translateX(3px);
        box-shadow: 0 5px 14px rgba(90, 101, 174, 0.12);
    }

    [data-testid="stSidebar"]
    div[role="radiogroup"]
    label:has(input:checked) {
        background: linear-gradient(
            90deg,
            #dde2ff,
            #e5f7f0
        );
        border: 1px solid #7278d8;
        box-shadow: 0 6px 15px rgba(90, 101, 174, 0.15);
        font-weight: 700;
    }

    [data-testid="stSidebar"] input[type="radio"] {
        accent-color: #666dd3;
    }

    [data-testid="stSidebar"] hr {
        border-color: #d7ddea;
    }

    .main-title {
        font-size: 48px;
        font-weight: 850;
        text-align: center;
        margin-bottom: 4px;
        letter-spacing: -1px;
        color: #27385c;
    }

    .title-highlight {
        color: #6d69d9;
    }

    .subtitle {
        text-align: center;
        color: #647087;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        color: #263755;
        font-size: 30px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 18px;
        padding-left: 14px;
        border-left: 5px solid #7975dd;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #dce2ed;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 28px rgba(55, 74, 109, 0.08);
    }

    .info-card h3 {
        color: #273755 !important;
    }

    .info-card p {
        color: #59667b !important;
    }

    .feature-card {
        background:
            linear-gradient(
                145deg,
                rgba(255, 255, 255, 0.97),
                rgba(244, 246, 255, 0.97)
            );
        border: 1px solid #dce1ee;
        border-top: 5px solid #7774dc;
        border-radius: 17px;
        padding: 20px;
        min-height: 175px;
        box-shadow: 0 8px 22px rgba(55, 74, 109, 0.07);
        transition: transform 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 13px 28px rgba(55, 74, 109, 0.11);
    }

    .feature-card h3 {
        color: #2c3c5d;
        margin-top: 0;
    }

    .feature-card p {
        color: #647086;
        line-height: 1.65;
    }

    .formula-box {
        background:
            linear-gradient(
                135deg,
                #eceeff,
                #e8f8f2
            );
        border: 1px solid #cfd5eb;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        font-size: 22px;
        font-weight: 650;
        color: #354365;
        margin-top: 15px;
        margin-bottom: 24px;
        box-shadow: 0 7px 20px rgba(55, 74, 109, 0.07);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #dce2ed;
        border-radius: 17px;
        padding: 18px;
        box-shadow: 0 8px 20px rgba(55, 74, 109, 0.07);
    }

    div[data-testid="stMetric"]:hover {
        border-color: #aaaee8;
        box-shadow: 0 11px 25px rgba(55, 74, 109, 0.10);
    }

    div[data-testid="stMetricLabel"] {
        color: #69758a !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #273755 !important;
        font-weight: 800;
    }

    h1, h2, h3, h4 {
        color: #273755;
    }

    p, li {
        color: #59667b;
    }

    [data-testid="stDataFrame"] {
        background: white;
        border: 1px solid #dce2ed;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 7px 18px rgba(55, 74, 109, 0.06);
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1rem;
        color: white;
        font-weight: 750;
        background:
            linear-gradient(
                90deg,
                #6f6bd8,
                #4fae9b
            );
        box-shadow: 0 7px 17px rgba(92, 97, 181, 0.22);
    }

    .stButton > button:hover {
        color: white;
        border: none;
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(92, 97, 181, 0.28);
    }

    .stSelectbox label,
    .stSlider label,
    .stToggle label {
        color: #34435f !important;
        font-weight: 650;
    }

    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #dce2ed;
        border-radius: 14px;
    }

    .status-success {
        padding: 15px 18px;
        background: #e8f8f1;
        border: 1px solid #a9dcc8;
        border-radius: 13px;
        color: #226c57;
        font-weight: 650;
    }

    div[data-testid="stAlert"] {
        border-radius: 13px;
    }

    .footer {
        text-align: center;
        color: #718096;
        margin-top: 42px;
        padding: 24px;
        border-top: 1px solid #d8deea;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.65);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MAZE ENVIRONMENT
# ---------------------------------------------------------

class MazeEnvironment:
    def __init__(self):
        self.maze = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 1, 0, 1, 0, 0],
            ]
        )

        self.rows, self.cols = self.maze.shape
        self.start_position = (0, 0)
        self.goal_position = (5, 5)
        self.current_position = self.start_position

        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }

    def reset(self):
        self.current_position = self.start_position
        return self.position_to_state(self.current_position)

    def position_to_state(self, position):
        row, col = position
        return row * self.cols + col

    def state_to_position(self, state):
        return state // self.cols, state % self.cols

    def step(self, action):
        row, col = self.current_position
        row_change, col_change = self.actions[action]

        new_row = row + row_change
        new_col = col + col_change

        if (
            new_row < 0
            or new_row >= self.rows
            or new_col < 0
            or new_col >= self.cols
        ):
            return (
                self.position_to_state(self.current_position),
                -10,
                False,
            )

        if self.maze[new_row, new_col] == 1:
            return (
                self.position_to_state(self.current_position),
                -10,
                False,
            )

        self.current_position = (new_row, new_col)

        if self.current_position == self.goal_position:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        return (
            self.position_to_state(self.current_position),
            reward,
            done,
        )


environment = MazeEnvironment()


# ---------------------------------------------------------
# LOAD SAVED PROJECT FILES
# ---------------------------------------------------------

@st.cache_data
def load_project_data():
    q_table_path = os.path.join("models", "q_table.npy")
    history_path = os.path.join("results", "training_history.csv")
    metrics_path = os.path.join("results", "evaluation_metrics.csv")
    optimal_path_file = os.path.join("results", "optimal_path.csv")

    required_files = {
        q_table_path: "Q-table",
        history_path: "training history",
        metrics_path: "evaluation metrics",
        optimal_path_file: "optimal path",
    }

    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The {description} file was not found: {file_path}"
            )

    q_table = np.load(q_table_path)
    training_history = pd.read_csv(history_path)
    evaluation_metrics = pd.read_csv(metrics_path)
    optimal_path_df = pd.read_csv(optimal_path_file)

    optimal_path = list(
        zip(
            optimal_path_df["Row"].astype(int),
            optimal_path_df["Column"].astype(int),
        )
    )

    return (
        q_table,
        training_history,
        evaluation_metrics,
        optimal_path,
    )


try:
    (
        q_table,
        training_history,
        evaluation_metrics,
        optimal_path,
    ) = load_project_data()

except FileNotFoundError as error:
    st.error(str(error))
    st.info(
        "Run the final notebook export cell and make sure the models "
        "and results folders are in the same folder as app.py."
    )
    st.stop()


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_metric_value(metric_name, default=0):
    matching_rows = evaluation_metrics[
        evaluation_metrics["Metric"] == metric_name
    ]

    if matching_rows.empty:
        return default

    return float(matching_rows.iloc[0]["Value"])


def apply_light_chart_theme(figure, height=450):
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        font={"color": "#34435f"},
        title_font={"color": "#273755", "size": 21},
        legend={
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": "#dce2ed",
            "borderwidth": 1,
        },
        margin=dict(l=50, r=30, t=70, b=50),
    )

    figure.update_xaxes(
        gridcolor="rgba(110,120,150,0.15)",
        zerolinecolor="rgba(110,120,150,0.20)",
        linecolor="#cfd6e4",
    )

    figure.update_yaxes(
        gridcolor="rgba(110,120,150,0.15)",
        zerolinecolor="rgba(110,120,150,0.20)",
        linecolor="#cfd6e4",
    )

    return figure


def create_maze_figure(path=None, agent_position=None):
    maze_values = np.zeros((environment.rows, environment.cols))
    maze_values[environment.maze == 1] = 1

    if path:
        for row, col in path:
            position = (row, col)

            if (
                position != environment.start_position
                and position != environment.goal_position
                and environment.maze[row, col] != 1
            ):
                maze_values[row, col] = 2

    start_row, start_col = environment.start_position
    goal_row, goal_col = environment.goal_position

    maze_values[start_row, start_col] = 3
    maze_values[goal_row, goal_col] = 4

    if agent_position is not None:
        agent_row, agent_col = agent_position

        if agent_position not in [
            environment.start_position,
            environment.goal_position,
        ]:
            maze_values[agent_row, agent_col] = 5

    text_grid = np.full(
        (environment.rows, environment.cols),
        "",
        dtype=object,
    )

    for row in range(environment.rows):
        for col in range(environment.cols):
            position = (row, col)

            if environment.maze[row, col] == 1:
                text_grid[row, col] = "Obstacle"
            elif position == environment.start_position:
                text_grid[row, col] = (
                    "Agent / Start"
                    if agent_position == position
                    else "Start"
                )
            elif position == environment.goal_position:
                text_grid[row, col] = (
                    "Agent / Goal"
                    if agent_position == position
                    else "Goal"
                )
            elif agent_position == position:
                text_grid[row, col] = "Agent"
            elif path and position in path:
                text_grid[row, col] = "Path"

    color_scale = [
        [0.00, "#f7f9fc"],
        [0.19, "#f7f9fc"],
        [0.20, "#6f7f9c"],
        [0.39, "#6f7f9c"],
        [0.40, "#f4c95d"],
        [0.59, "#f4c95d"],
        [0.60, "#77c9b5"],
        [0.79, "#77c9b5"],
        [0.80, "#86d79d"],
        [0.94, "#86d79d"],
        [0.95, "#ef7d8a"],
        [1.00, "#ef7d8a"],
    ]

    figure = go.Figure(
        data=go.Heatmap(
            z=maze_values,
            text=text_grid,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#1f2d42"},
            colorscale=color_scale,
            showscale=False,
            x=list(range(environment.cols)),
            y=list(range(environment.rows)),
            hovertemplate=(
                "Row: %{y}<br>"
                "Column: %{x}<br>"
                "%{text}<extra></extra>"
            ),
            xgap=2,
            ygap=2,
        )
    )

    figure.update_layout(
        height=590,
        margin=dict(l=30, r=30, t=65, b=35),
        title={
            "text": "Q-Learning Maze Environment",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "#273755", "size": 22},
        },
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
        font={"color": "#34435f"},
        xaxis={
            "title": "Column",
            "side": "bottom",
            "dtick": 1,
            "showgrid": False,
            "linecolor": "#cfd6e4",
        },
        yaxis={
            "title": "Row",
            "autorange": "reversed",
            "dtick": 1,
            "showgrid": False,
            "scaleanchor": "x",
            "scaleratio": 1,
            "linecolor": "#cfd6e4",
        },
    )

    return figure


def create_policy_dataframe():
    symbols = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→",
    }

    policy_grid = np.full(
        (environment.rows, environment.cols),
        "",
        dtype=object,
    )

    for row in range(environment.rows):
        for col in range(environment.cols):
            position = (row, col)

            if environment.maze[row, col] == 1:
                policy_grid[row, col] = "■"

            elif position == environment.goal_position:
                policy_grid[row, col] = "G"

            else:
                state = environment.position_to_state(position)
                best_action = int(np.argmax(q_table[state]))

                if position == environment.start_position:
                    policy_grid[row, col] = (
                        "S " + symbols[best_action]
                    )
                else:
                    policy_grid[row, col] = symbols[best_action]

    return pd.DataFrame(
        policy_grid,
        columns=[
            f"Column {index}"
            for index in range(environment.cols)
        ],
        index=[
            f"Row {index}"
            for index in range(environment.rows)
        ],
    )


def get_q_table_dataframe():
    dataframe = pd.DataFrame(
        q_table,
        columns=["Up", "Down", "Left", "Right"],
    )
    dataframe.index.name = "State"
    return dataframe


def create_q_value_heatmap():
    q_dataframe = get_q_table_dataframe()
    state_labels = []

    for state in range(len(q_dataframe)):
        position = environment.state_to_position(state)

        if environment.maze[position] == 1:
            state_labels.append(f"{state} — Obstacle")
        elif position == environment.start_position:
            state_labels.append(f"{state} — Start")
        elif position == environment.goal_position:
            state_labels.append(f"{state} — Goal")
        else:
            state_labels.append(str(state))

    figure = px.imshow(
        q_dataframe.values,
        labels={
            "x": "Action",
            "y": "State",
            "color": "Q-Value",
        },
        x=q_dataframe.columns,
        y=state_labels,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Tealgrn",
    )

    figure.update_traces(
        hovertemplate=(
            "State: %{y}<br>"
            "Action: %{x}<br>"
            "Q-Value: %{z:.3f}<extra></extra>"
        )
    )

    figure.update_layout(
        height=980,
        title="Learned Q-Values for Every State and Action",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.8)",
        font={"color": "#34435f"},
        title_font={"color": "#273755", "size": 21},
        margin=dict(l=80, r=40, t=70, b=50),
        coloraxis_colorbar={
            "title": "Q-Value",
            "tickfont": {"color": "#34435f"},
            "titlefont": {"color": "#34435f"},
        },
    )

    return figure


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #e4e7ff, #e3f7ef);
        border: 1px solid #d4daeb;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 7px 18px rgba(55, 74, 109, 0.08);
    ">
        <div style="font-size:38px;">🧠</div>
        <div style="
            font-size:24px;
            font-weight:800;
            color:#263755;
        ">
            MazeMind
        </div>
        <div style="
            font-size:13px;
            color:#68758a;
            margin-top:4px;
        ">
            Reinforcement Learning Lab
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Choose a dashboard section",
    [
        "🏠 Home",
        "🧩 Maze Environment",
        "📈 Training Analytics",
        "🧭 Optimal Policy",
        "🔢 Q-Table Explorer",
        "📘 About Project",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Training Setup")

st.sidebar.markdown(
    """
    **Algorithm:** Q-Learning  
    **Maze Size:** 6 × 6  
    **Total States:** 36  
    **Actions:** 4  
    **Episodes:** 2,000  
    **Learning Rate:** 0.10  
    **Discount Factor:** 0.95  
    **Final Epsilon:** 0.01
    """
)

st.sidebar.markdown("---")
st.sidebar.success("✓ Trained model loaded")


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="main-title">
        Maze<span class="title-highlight">Mind</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        A visual reinforcement learning laboratory for intelligent
        maze navigation
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

if page == "🏠 Home":
    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h3>Intelligent Q-Learning Maze Solver</h3>
            <p>
                This project demonstrates how an autonomous agent learns
                to navigate through a maze using reinforcement learning.
                The agent receives a positive reward for reaching the goal,
                penalties for invalid movements and a small step penalty
                that encourages it to discover the shortest valid path.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Overall Success",
        f"{get_metric_value('Overall Success Rate (%)'):.2f}%",
    )
    col2.metric(
        "Last 100 Success",
        f"{get_metric_value('Last 100 Success Rate (%)'):.2f}%",
    )
    col3.metric(
        "Optimal Path",
        f"{int(get_metric_value('Optimal Path Moves'))} moves",
    )
    col4.metric(
        "Final Epsilon",
        f"{get_metric_value('Final Epsilon'):.2f}",
    )

    st.markdown("### How the Agent Learns")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>1. Explore</h3>
                <p>
                    The agent initially selects many random actions to
                    understand the maze and discover possible routes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>2. Learn</h3>
                <p>
                    Rewards and penalties update Q-values that represent
                    the usefulness of each action in every state.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>3. Exploit</h3>
                <p>
                    After training, the agent selects actions with the
                    highest Q-values and follows the learned optimal path.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Q-Learning Update Equation")

    st.markdown(
        """
        <div class="formula-box">
            Q(s, a) ← Q(s, a) + α
            [r + γ max Q(s′, a′) − Q(s, a)]
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Reinforcement Learning Workflow")

    workflow = pd.DataFrame(
        {
            "Stage": [
                "Initialize Environment",
                "Select Action",
                "Execute Movement",
                "Receive Reward",
                "Update Q-Table",
                "Learn Optimal Policy",
            ],
            "Description": [
                "Create the maze, start state and goal state.",
                "Use epsilon-greedy exploration and exploitation.",
                "Move the agent according to the selected action.",
                "Apply rewards, step penalties or obstacle penalties.",
                "Update knowledge using the Bellman equation.",
                "Repeat training until the shortest path is learned.",
            ],
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# MAZE ENVIRONMENT PAGE
# ---------------------------------------------------------

elif page == "🧩 Maze Environment":
    st.markdown(
        '<div class="section-title">Maze Environment</div>',
        unsafe_allow_html=True,
    )

    info1, info2, info3, info4 = st.columns(4)

    info1.metric("Grid Size", "6 × 6")
    info2.metric(
        "Valid States",
        int(np.sum(environment.maze == 0)),
    )
    info3.metric(
        "Obstacles",
        int(np.sum(environment.maze == 1)),
    )
    info4.metric("Available Actions", 4)

    show_path = st.toggle(
        "Display learned optimal path",
        value=False,
    )

    selected_path = optimal_path if show_path else None

    st.plotly_chart(
        create_maze_figure(path=selected_path),
        use_container_width=True,
    )

    legend1, legend2, legend3, legend4 = st.columns(4)

    legend1.info("Start: Agent's initial state")
    legend2.success("Goal: Final destination")
    legend3.warning("Path: Learned optimal route")
    legend4.error("Obstacle: Invalid state")

    st.markdown("### Reward Structure")

    rewards = pd.DataFrame(
        {
            "Event": [
                "Reach Goal",
                "Valid Movement",
                "Hit Obstacle",
                "Hit Boundary",
            ],
            "Reward": [
                "+100",
                "-1",
                "-10",
                "-10",
            ],
            "Purpose": [
                "Encourages the agent to reach the destination.",
                "Encourages shorter paths.",
                "Discourages invalid movements.",
                "Keeps the agent inside the maze.",
            ],
        }
    )

    st.dataframe(
        rewards,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# TRAINING ANALYTICS PAGE
# ---------------------------------------------------------

elif page == "📈 Training Analytics":
    st.markdown(
        '<div class="section-title">Training Analytics</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Episodes",
        int(get_metric_value("Total Episodes")),
    )
    metric2.metric(
        "Average Reward",
        f"{get_metric_value('Overall Average Reward'):.2f}",
    )
    metric3.metric(
        "Average Steps",
        f"{get_metric_value('Overall Average Steps'):.2f}",
    )
    metric4.metric(
        "Last 100 Steps",
        f"{get_metric_value('Last 100 Average Steps'):.2f}",
    )

    window_size = st.slider(
        "Moving-average window",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    analytics_data = training_history.copy()

    analytics_data["Reward Moving Average"] = (
        analytics_data["Reward"]
        .rolling(window=window_size)
        .mean()
    )

    analytics_data["Steps Moving Average"] = (
        analytics_data["Steps"]
        .rolling(window=window_size)
        .mean()
    )

    reward_figure = go.Figure()

    reward_figure.add_trace(
        go.Scatter(
            x=analytics_data["Episode"],
            y=analytics_data["Reward"],
            mode="lines",
            name="Episode Reward",
            opacity=0.35,
            line={"color": "#8ea4d8"},
        )
    )

    reward_figure.add_trace(
        go.Scatter(
            x=analytics_data["Episode"],
            y=analytics_data["Reward Moving Average"],
            mode="lines",
            name=f"{window_size}-Episode Moving Average",
            line={"width": 3, "color": "#6f6bd8"},
        )
    )

    reward_figure.update_layout(
        title="Reward Improvement Across Episodes",
        xaxis_title="Episode",
        yaxis_title="Total Reward",
        hovermode="x unified",
    )

    apply_light_chart_theme(reward_figure, height=460)

    st.plotly_chart(
        reward_figure,
        use_container_width=True,
    )

    step_figure = go.Figure()

    step_figure.add_trace(
        go.Scatter(
            x=analytics_data["Episode"],
            y=analytics_data["Steps"],
            mode="lines",
            name="Steps per Episode",
            opacity=0.35,
            line={"color": "#a7b7d9"},
        )
    )

    step_figure.add_trace(
        go.Scatter(
            x=analytics_data["Episode"],
            y=analytics_data["Steps Moving Average"],
            mode="lines",
            name=f"{window_size}-Episode Moving Average",
            line={"width": 3, "color": "#4fae9b"},
        )
    )

    step_figure.update_layout(
        title="Steps Taken by the Agent per Episode",
        xaxis_title="Episode",
        yaxis_title="Number of Steps",
        hovermode="x unified",
    )

    apply_light_chart_theme(step_figure, height=460)

    st.plotly_chart(
        step_figure,
        use_container_width=True,
    )

    epsilon_figure = px.line(
        analytics_data,
        x="Episode",
        y="Epsilon",
        title="Exploration Rate Decay",
    )

    epsilon_figure.update_traces(
        line={"width": 3, "color": "#7b78df"},
    )

    epsilon_figure.update_layout(
        xaxis_title="Episode",
        yaxis_title="Epsilon",
    )

    apply_light_chart_theme(epsilon_figure, height=430)

    st.plotly_chart(
        epsilon_figure,
        use_container_width=True,
    )

    success_by_block = (
        analytics_data
        .assign(
            Episode_Block=(
                (analytics_data["Episode"] - 1) // 100 + 1
            )
        )
        .groupby("Episode_Block", as_index=False)["Success"]
        .mean()
    )

    success_by_block["Success Rate (%)"] = (
        success_by_block["Success"] * 100
    )

    success_figure = px.bar(
        success_by_block,
        x="Episode_Block",
        y="Success Rate (%)",
        title="Success Rate for Every 100 Episodes",
        color="Success Rate (%)",
        color_continuous_scale="Mint",
    )

    success_figure.update_layout(
        xaxis_title="100-Episode Block",
        yaxis_title="Success Rate (%)",
        coloraxis_showscale=False,
    )

    apply_light_chart_theme(success_figure, height=430)

    st.plotly_chart(
        success_figure,
        use_container_width=True,
    )


# ---------------------------------------------------------
# OPTIMAL POLICY PAGE
# ---------------------------------------------------------

elif page == "🧭 Optimal Policy":
    st.markdown(
        '<div class="section-title">Learned Optimal Policy</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Optimal Path Length",
        f"{len(optimal_path) - 1} moves",
    )
    metric2.metric(
        "Last 100 Success",
        f"{get_metric_value('Last 100 Success Rate (%)'):.2f}%",
    )
    metric3.metric(
        "Last 100 Reward",
        f"{get_metric_value('Last 100 Average Reward'):.2f}",
    )

    st.markdown(
        """
        <div class="status-success">
            The learned policy reaches the goal successfully using the
            shortest discovered path.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    animate_agent = st.button(
        "▶ Animate Agent Movement",
        use_container_width=True,
    )

    animation_placeholder = st.empty()

    if animate_agent:
        for step_number, position in enumerate(optimal_path):
            with animation_placeholder.container():
                st.plotly_chart(
                    create_maze_figure(
                        path=optimal_path,
                        agent_position=position,
                    ),
                    use_container_width=True,
                    key=f"animation_{step_number}",
                )

                st.caption(
                    f"Step {step_number}: Agent position {position}"
                )

            time.sleep(0.55)

        st.success(
            "The agent successfully reached the goal."
        )

    else:
        animation_placeholder.plotly_chart(
            create_maze_figure(path=optimal_path),
            use_container_width=True,
        )

    st.markdown("### Optimal Path Coordinates")

    path_dataframe = pd.DataFrame(
        optimal_path,
        columns=["Row", "Column"],
    )

    path_dataframe.insert(
        0,
        "Step",
        range(len(path_dataframe)),
    )

    st.dataframe(
        path_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Learned Policy Map")

    st.dataframe(
        create_policy_dataframe(),
        use_container_width=True,
    )

    st.caption(
        "↑ Up | ↓ Down | ← Left | → Right | "
        "■ Obstacle | S Start | G Goal"
    )


# ---------------------------------------------------------
# Q-TABLE EXPLORER PAGE
# ---------------------------------------------------------

elif page == "🔢 Q-Table Explorer":
    st.markdown(
        '<div class="section-title">Q-Table Explorer</div>',
        unsafe_allow_html=True,
    )

    q_dataframe = get_q_table_dataframe()

    st.plotly_chart(
        create_q_value_heatmap(),
        use_container_width=True,
    )

    st.markdown("### Inspect an Individual State")

    selected_state = st.selectbox(
        "Select state number",
        options=list(
            range(environment.rows * environment.cols)
        ),
    )

    selected_position = environment.state_to_position(
        selected_state
    )

    selected_values = q_dataframe.loc[selected_state]

    if environment.maze[selected_position] == 1:
        state_type = "Obstacle"
    elif selected_position == environment.start_position:
        state_type = "Start State"
    elif selected_position == environment.goal_position:
        state_type = "Goal State"
    else:
        state_type = "Valid State"

    col1, col2, col3 = st.columns(3)

    col1.metric("State", selected_state)
    col2.metric("Position", str(selected_position))
    col3.metric("State Type", state_type)

    selected_q_values = pd.DataFrame(
        {
            "Action": selected_values.index,
            "Q-Value": selected_values.values,
        }
    )

    state_figure = px.bar(
        selected_q_values,
        x="Action",
        y="Q-Value",
        title=f"Action Values for State {selected_state}",
        color="Action",
        color_discrete_sequence=[
            "#7b78df",
            "#4fae9b",
            "#e0a84e",
            "#e47b91",
        ],
    )

    state_figure.update_layout(
        showlegend=False,
    )

    apply_light_chart_theme(state_figure, height=420)

    st.plotly_chart(
        state_figure,
        use_container_width=True,
    )

    if state_type in ["Valid State", "Start State"]:
        best_action = selected_values.idxmax()
        best_value = selected_values.max()

        st.success(
            f"Best learned action: {best_action} "
            f"with Q-value {best_value:.2f}"
        )

    elif state_type == "Obstacle":
        st.warning(
            "This state represents an obstacle. The agent cannot "
            "enter this cell, so its Q-values remain zero."
        )

    else:
        st.info(
            "This is the terminal goal state. No further action "
            "is required after reaching it."
        )

    with st.expander("View complete Q-table"):
        st.dataframe(
            q_dataframe.style.format("{:.3f}"),
            use_container_width=True,
        )


# ---------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------

elif page == "📘 About Project":
    st.markdown(
        '<div class="section-title">About the Project</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h3>Project Objective</h3>
            <p>
                The objective of this project is to develop an
                autonomous maze-solving agent using Q-Learning. The
                project demonstrates how reinforcement learning can
                make decisions through repeated interaction with an
                environment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Technologies Used")

    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "NumPy",
                "Pandas",
                "Q-Learning",
                "Plotly",
                "Streamlit",
            ],
            "Usage": [
                "Application development",
                "Q-table and numerical operations",
                "Training data analysis",
                "Reinforcement learning algorithm",
                "Interactive visualizations",
                "Web dashboard and deployment",
            ],
        }
    )

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Important Hyperparameters")

    parameters = pd.DataFrame(
        {
            "Parameter": [
                "Learning Rate",
                "Discount Factor",
                "Initial Epsilon",
                "Minimum Epsilon",
                "Epsilon Decay",
                "Episodes",
                "Maximum Steps",
            ],
            "Value": [
                "0.10",
                "0.95",
                "1.00",
                "0.01",
                "0.995",
                "2,000",
                "200",
            ],
            "Purpose": [
                "Controls the speed of Q-value updates.",
                "Controls the importance of future rewards.",
                "Starts with maximum exploration.",
                "Maintains minimal random exploration.",
                "Gradually reduces exploration.",
                "Number of complete training attempts.",
                "Prevents an episode from running indefinitely.",
            ],
        }
    )

    st.dataframe(
        parameters,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Real-World Applications")

    application1, application2, application3 = st.columns(3)

    with application1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>Robotics</h3>
                <p>
                    Autonomous robots can learn safe and efficient
                    navigation paths in warehouses or indoor spaces.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with application2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>Game Intelligence</h3>
                <p>
                    Game agents can learn strategies through rewards,
                    penalties and repeated interaction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with application3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>Route Optimization</h3>
                <p>
                    Reinforcement learning can help discover efficient
                    routes under changing environmental conditions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Future Enhancements")

    st.markdown(
        """
        - Support user-designed mazes and dynamic obstacles.
        - Compare Q-Learning with SARSA and Deep Q-Networks.
        - Add real-time training controls to the dashboard.
        - Introduce larger and more complex environments.
        - Add multiple agents and competitive navigation.
        """
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Developed as part of the Machine Learning Internship Project
        <br>
        <b>Deep Reinforcement Learning using Q-Learning</b>
    </div>
    """,
    unsafe_allow_html=True,
)
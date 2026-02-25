import sys

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

replacement = '''
def _set_style() -> None:
    st.markdown(
        """
        <style>
            :root {
                --apris-border-soft: rgba(226, 232, 240, 0.8);
                --background-color: #f8fafc;
                --secondary-background-color: #ffffff;
                --text-color: #0f172a;
                --primary-color: #6366f1; /* Indigo */
                --primary-hover: #4f46e5;
                --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
                --card-shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
            }
            /* Global Font Settings */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, .stApp {
                font-family: 'Inter', sans-serif !important;
                color: var(--text-color) !important;
                background-color: var(--background-color) !important;
                -webkit-font-smoothing: antialiased;
            }
            /* Clean up background */
            .stApp {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
            }
            .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp span,
            [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], .stMarkdown, .stCaption {
                color: var(--text-color) !important;
            }
            /* Headings */
            .stApp h1 {
                font-weight: 800 !important;
                letter-spacing: -0.025em;
                background: linear-gradient(90deg, #312e81, #6366f1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            .stApp h2, .stApp h3, .stApp h4 {
                font-weight: 700 !important;
                letter-spacing: -0.015em;
                color: #1e293b !important;
            }
            /* Header and navigation clean up */
            header[data-testid="stHeader"] {
                background: transparent !important;
                box-shadow: none !important;
            }
            [data-testid="stToolbar"], [data-testid="collapsedControl"], [data-testid="stSidebar"] {
                display: none !important;
            }
            /* Buttons */
            div.stButton > button, div.stFormSubmitButton > button {
                background: var(--secondary-background-color);
                color: #334155;
                border: 1px solid var(--apris-border-soft);
                border-radius: 12px !important;
                font-weight: 600 !important;
                min-height: 48px !important;
                padding: 0 1.5rem !important;
                transition: all 0.2s ease-in-out !important;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            }
            div.stButton > button:hover, div.stFormSubmitButton > button:hover {
                border-color: #cbd5e1 !important;
                background-color: #f8fafc !important;
                color: #0f172a !important;
                transform: translateY(-1px);
                box-shadow: var(--card-shadow);
            }
            div[data-testid="stFormSubmitButton"] button[kind="primary"],
            div.stButton > button[kind="primary"] {
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%) !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.4), 0 2px 4px -2px rgba(99, 102, 241, 0.4) !important;
            }
            div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
            div.stButton > button[kind="primary"]:hover {
                box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.5), 0 4px 6px -4px rgba(99, 102, 241, 0.5) !important;
                transform: translateY(-2px);
            }
            div[data-testid="stFormSubmitButton"] button[kind="primary"]:active,
            div.stButton > button[kind="primary"]:active {
                transform: translateY(0);
            }
            /* Metrics Cards */
            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.5);
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: var(--card-shadow);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            [data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                box-shadow: var(--card-shadow-hover);
            }
            [data-testid="stMetricLabel"] {
                color: #64748b !important;
                font-size: 0.875rem !important;
                font-weight: 500 !important;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.25rem;
            }
            [data-testid="stMetricValue"] {
                color: #0f172a !important;
                font-weight: 800 !important;
                font-size: 2rem !important;
                line-height: 1 !important;
            }
            /* Tabs styling */
            [data-baseweb="tab-list"] {
                gap: 1rem;
                border-bottom: 2px solid #e2e8f0;
            }
            [data-baseweb="tab"] {
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
            }
            [data-baseweb="tab"] p {
                font-weight: 600;
                font-size: 1.05rem;
                color: #64748b !important;
            }
            [aria-selected="true"] p {
                color: var(--primary-color) !important;
            }
            [data-baseweb="tab-highlight"] {
                background-color: var(--primary-color) !important;
                height: 3px !important;
                border-radius: 3px 3px 0 0 !important;
            }
            [data-baseweb="tab-panel"] {
                background: transparent;
                padding: 1.5rem 0;
            }
            /* Data tables */
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                border: none !important;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: var(--card-shadow);
                background: white;
            }
            /* Main Content Cards */
            .apris-card {
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.6);
                border-radius: 20px;
                padding: 1.5rem 2rem;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
                margin-bottom: 1.5rem;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .apris-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            }
            /* Input fields styling */
            [data-baseweb="input"], [data-baseweb="base-input"] {
                background: #f8fafc !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 12px !important;
                transition: all 0.2s ease;
            }
            [data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within {
                border-color: var(--primary-color) !important;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
                background: #ffffff !important;
            }
            [data-baseweb="input"] input, [data-baseweb="base-input"] input {
                color: #0f172a !important;
                -webkit-text-fill-color: #0f172a !important;
                font-weight: 500 !important;
                font-size: 1rem !important;
            }
            /* Risk banners */
            .risk-banner {
                border-radius: 16px;
                padding: 1rem 1.5rem;
                margin-bottom: 1.5rem;
                font-weight: 700;
                font-size: 1.125rem;
                display: flex;
                align-items: center;
                box-shadow: var(--card-shadow);
            }
            .risk-banner::before {
                content: '';
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 12px;
            }
            .risk-banner-low {
                background: linear-gradient(to right, #ecfdf5, #d1fae5);
                color: #065f46 !important;
                border: 1px solid #10b981;
            }
            .risk-banner-low::before { background-color: #10b981; box-shadow: 0 0 8px #10b981; }
            .risk-banner-medium {
                background: linear-gradient(to right, #fffbeb, #fef3c7);
                color: #b45309 !important;
                border: 1px solid #f59e0b;
            }
            .risk-banner-medium::before { background-color: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
            .risk-banner-high {
                background: linear-gradient(to right, #fef2f2, #fee2e2);
                color: #b91c1c !important;
                border: 1px solid #ef4444;
            }
            .risk-banner-high::before { background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }
            
            /* Expanders */
            [data-testid="stExpander"] {
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                background: white;
                box-shadow: var(--card-shadow);
                overflow: hidden;
            }
            [data-testid="stExpander"] > details > summary {
                padding: 1rem 1.5rem;
                font-weight: 600;
                background: #f8fafc;
            }
            [data-testid="stExpander"] > details > summary:hover {
                background: #f1f5f9;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _apply_matplotlib_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "none",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#E2E8F0",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.color": "#334155",
            "axes.titleweight": "bold",
            "font.size": 10,
            "text.color": "#0F172A",
            "axes.labelcolor": "#0F172A",
            "xtick.color": "#0F172A",
            "ytick.color": "#0F172A",
        }
    )
'''

# We only want to keep lines up to 163 (inclusive of index 162), 
# because 164 is where the bad duplication starts.
# Then we continue from line 528 (index 527) upwards.
new_lines = lines[:163]

# Remove leading newline from replacement when joining
if replacement.startswith("\\n"):
    replacement = replacement[1:]

new_lines.append(replacement)

if len(lines) > 527:
    # Ensure line 528 starts properly
    if not lines[527].startswith("\\n"):
        new_lines.append("\\n")
    new_lines.extend(lines[527:])

with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("app.py fixed successfully.")

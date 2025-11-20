# utils.py
import streamlit as st

def load_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" />

    <style>
        /* Global Font Application */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Animation Keyframes */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translate3d(0, 20px, 0); }
            to { opacity: 1; transform: translate3d(0, 0, 0); }
        }

        /* Main Header Styling */
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #1f77b4, #00C9FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            animation: fadeInUp 0.8s ease-out;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
            animation: fadeInUp 0.8s ease-out;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
            border-color: #1f77b4;
        }

        /* Insight Box */
        .insight-box {
            background: rgba(31, 119, 180, 0.08);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border-left: 4px solid #1f77b4;
            color: var(--text-color);
            animation: fadeInUp 1s ease-out;
        }
        
        .insight-box h4 {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #1f77b4;
        }

        .insight-box ul {
            list-style-type: none;
            padding-left: 0;
        }

        .insight-box li {
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Icons */
        .material-symbols-rounded {
            font-size: 24px;
            vertical-align: middle;
        }
    </style>
    """, unsafe_allow_html=True)
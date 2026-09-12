import os
import io
import time
import math
import pickle
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="InspireAI - Advanced LSTM Quote Generator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Generator (Ultra-Modern Glassmorphism Soft Light & Premium Dark Themes)
# ---------------------------------------------------------
def get_theme_css(theme_name):
    light_css = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
    }

    div[data-testid="stDecoration"] { display: none !important; }

    header[data-testid="stHeader"], [data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    button[data-testid="baseButton-header"],
    button[data-testid="baseButton-headerNoPadding"],
    button[aria-label*="sidebar"],
    button[aria-label*="Sidebar"],
    header[data-testid="stHeader"] button {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #0f172a !important;
        fill: #0f172a !important;
        border-radius: 10px !important;
        border: 1.5px solid #cbd5e1 !important;
    }

    header[data-testid="stHeader"] svg, button[aria-label*="sidebar"] svg {
        fill: #0f172a !important;
        color: #0f172a !important;
    }

    /* White / Light Mode Background */
    .stApp {
        background: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
            linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%);
        background-attachment: fixed;
        color: #0f172a !important;
    }

    /* Global Text & Label Overrides for Light Mode */
    .stApp, .stApp p, .stApp label,
    div[data-testid="stMarkdownContainer"] *:not(.flowchart-container *):not(.flowchart-card-title):not(.flowchart-card-sub):not(.flowchart-card-exp),
    div[data-testid="stWidgetLabel"] * {
        color: #0f172a !important;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #0f172a !important;
    }

    .flowchart-container *, .flowchart-card-title, .flowchart-card-title * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .stCaption, .stCaption p, .stCaption span {
        color: #475569 !important;
    }

    /* Sidebar for Light Mode */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.03);
    }
    
    section[data-testid="stSidebar"] *, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] strong {
        color: #0f172a !important;
    }

    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #1e1b4b !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
    }

    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        font-size: 3.2rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 35%, #7c3aed 70%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    
    .sub-title {
        font-size: 1.18rem;
        color: #475569 !important;
        font-weight: 500;
        margin-bottom: 1.8rem;
        line-height: 1.6;
    }

    /* Input & Selectbox & Textarea Styling for Light Mode */
    .stTextInput > div > div,
    div[data-baseweb="input"],
    .stSelectbox > div > div,
    div[data-baseweb="select"],
    .stTextArea > div > div,
    div[data-baseweb="textarea"],
    div[data-baseweb="textarea"] textarea,
    textarea {
        border-radius: 12px !important;
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
        color: #0f172a !important;
    }

    .stTextInput input,
    div[data-baseweb="input"] input,
    input[type="text"],
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] span,
    .stTextArea textarea,
    div[data-baseweb="textarea"] textarea,
    textarea {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Popover & Selectbox Dropdown Menu for Light Mode */
    ul[role="listbox"], div[data-baseweb="popover"] *, div[data-baseweb="menu"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    li[role="option"]:hover {
        background-color: #e0e7ff !important;
    }

    /* Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 10px;
        margin-bottom: 10px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(8px);
        color: #334155 !important;
        border: 1px solid rgba(203, 213, 225, 0.8);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }

    .badge-pill-purple { background: linear-gradient(135deg, rgba(238, 242, 255, 0.95) 0%, rgba(224, 231, 255, 0.95) 100%); color: #4338ca !important; border: 1px solid #c7d2fe; }
    .badge-pill-emerald { background: linear-gradient(135deg, rgba(236, 253, 245, 0.95) 0%, rgba(209, 250, 229, 0.95) 100%); color: #047857 !important; border: 1px solid #a7f3d0; }
    .badge-pill-gold { background: linear-gradient(135deg, rgba(255, 251, 235, 0.95) 0%, rgba(254, 243, 199, 0.95) 100%); color: #b45309 !important; border: 1px solid #fde68a; }

    /* Quote Card Container */
    .official-card-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(224, 231, 255, 0.9);
        border-radius: 24px;
        padding: 3.8rem 2.8rem;
        margin-top: 1.8rem;
        position: relative;
        text-align: center;
        color: #0f172a !important;
        box-shadow: 0 25px 60px -15px rgba(79, 70, 229, 0.12);
    }

    .official-badge { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #ffffff !important; padding: 7px 24px; border-radius: 30px; font-size: 0.8rem; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1.5rem; display: inline-block; }
    .official-quote-mark { font-family: 'Playfair Display', serif; font-size: 6rem; line-height: 0.2; color: #6366f1 !important; display: block; margin: 1.2rem auto 1.4rem auto; }
    .official-quote-text { font-size: 2.25rem; font-weight: 800; line-height: 1.5; margin: 1.5rem 0; color: #0f172a !important; }
    .official-quote-line-accent { color: #4338ca !important; font-weight: 800; font-size: 2.45rem; letter-spacing: 0.5px; display: block; margin: 6px 0; }
    .official-quote-line-plain { color: #1e293b !important; font-size: 2.05rem; font-weight: 700; display: block; margin: 6px 0; }
    .official-divider { width: 140px; height: 4px; background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%); margin: 2rem auto; border-radius: 4px; }
    .official-author { font-weight: 800; font-size: 1.15rem; letter-spacing: 2px; color: #334155 !important; text-transform: uppercase; margin-top: 1rem; }
    .official-handle { font-size: 0.88rem; color: #64748b !important; margin-top: 6px; font-weight: 500; }

    /* All Buttons & Download Buttons for Light Mode */
    .stButton > button,
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button,
    .stDownloadButton > button,
    button[data-testid="stBaseButton-secondary"],
    button[kind="secondary"],
    button[data-baseweb="button"] {
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.2rem !important;
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border: 1.5px solid #cbd5e1 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button *,
    div[data-testid="stButton"] > button *,
    div[data-testid="stDownloadButton"] > button *,
    div[data-testid="stFormSubmitButton"] > button *,
    .stDownloadButton > button *,
    button[data-testid="stBaseButton-secondary"] *,
    button[kind="secondary"] * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    .stButton > button:hover,
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    .stDownloadButton > button:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        background-color: #4f46e5 !important;
        border-color: #4f46e5 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton > button:hover *,
    div[data-testid="stButton"] > button:hover *,
    div[data-testid="stDownloadButton"] > button:hover *,
    div[data-testid="stFormSubmitButton"] > button:hover *,
    .stDownloadButton > button:hover *,
    button[data-testid="stBaseButton-secondary"]:hover * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stDownloadButton"] > button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 800 !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] *,
    div[data-testid="stDownloadButton"] > button[kind="primary"] *,
    button[data-testid="stBaseButton-primary"] *,
    button[kind="primary"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Reset Stats Button Red Danger Styling for Light Mode */
    .stButton > button[key="quiz_reset_stats_btn"],
    div[data-testid="stButton"] > button[key="quiz_reset_stats_btn"],
    button[key="quiz_reset_stats_btn"] {
        background-color: #fef2f2 !important;
        background: #fef2f2 !important;
        border: 1.5px solid #ef4444 !important;
        color: #dc2626 !important;
        -webkit-text-fill-color: #dc2626 !important;
        font-weight: 800 !important;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.15) !important;
    }
    .stButton > button[key="quiz_reset_stats_btn"]:hover,
    div[data-testid="stButton"] > button[key="quiz_reset_stats_btn"]:hover,
    button[key="quiz_reset_stats_btn"]:hover {
        background-color: #dc2626 !important;
        background: #dc2626 !important;
        border-color: #b91c1c !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 10px; background: rgba(241, 245, 249, 0.8); padding: 6px; border-radius: 14px; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 10px 20px; color: #64748b !important; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #ffffff !important; color: #4f46e5 !important; font-weight: 700 !important; }

    /* Expander & File Uploader Contrast Fixes for Light Mode */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 14px !important;
        margin-bottom: 14px !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stExpander"] summary * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    div[data-testid="stFileUploader"],
    div[data-testid="stFileUploader"] > section,
    section[data-testid="stFileUploadDropzone"],
    div[data-baseweb="file-uploader"],
    div[data-baseweb="file-uploader"] > div {
        background-color: #ffffff !important;
        border: 1.5px dashed #4f46e5 !important;
        border-radius: 14px !important;
        padding: 16px !important;
    }
    div[data-testid="stFileUploader"] *,
    section[data-testid="stFileUploadDropzone"] * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    div[data-testid="stFileUploader"] button,
    section[data-testid="stFileUploadDropzone"] button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        border: none !important;
        padding: 8px 18px !important;
    }
    """

    dark_css = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    div[data-testid="stDecoration"] { display: none !important; }

    header[data-testid="stHeader"], [data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    button[data-testid="baseButton-header"],
    button[data-testid="baseButton-headerNoPadding"],
    button[aria-label*="sidebar"],
    button[aria-label*="Sidebar"],
    header[data-testid="stHeader"] button {
        background-color: rgba(15, 23, 42, 0.85) !important;
        color: #38bdf8 !important;
        fill: #38bdf8 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }

    header[data-testid="stHeader"] svg, button[aria-label*="sidebar"] svg {
        fill: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    .stApp {
        background: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.18) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.18) 0px, transparent 50%),
            radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
        background-attachment: fixed;
        color: #f8fafc !important;
    }

    /* Global Text & Labels for Dark Theme */
    .stApp, .stApp p, .stApp label,
    div[data-testid="stMarkdownContainer"] *,
    div[data-testid="stWidgetLabel"] * {
        color: #f8fafc !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #f8fafc !important;
    }
    .stCaption, .stCaption p, .stCaption span {
        color: #93c5fd !important;
    }

    /* Sidebar for Dark Theme */
    section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(56, 189, 248, 0.3);
    }
    section[data-testid="stSidebar"] *, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] strong {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #38bdf8 !important; font-family: 'Space Grotesk', sans-serif;
    }

    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 900; font-size: 3.2rem;
        background: linear-gradient(135deg, #60a5fa 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem; letter-spacing: -1px;
    }
    .sub-title { color: #93c5fd !important; font-size: 1.18rem; margin-bottom: 1.8rem; }

    /* Inputs, Selectboxes, & Textareas for Dark Theme */
    .stTextInput > div > div,
    div[data-baseweb="input"],
    .stSelectbox > div > div,
    div[data-baseweb="select"],
    .stTextArea > div > div,
    div[data-baseweb="textarea"],
    div[data-baseweb="textarea"] textarea,
    textarea {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border: 1.5px solid rgba(56, 189, 248, 0.5) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }
    .stTextInput input,
    div[data-baseweb="input"] input,
    input[type="text"],
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] span,
    .stTextArea textarea,
    div[data-baseweb="textarea"] textarea,
    textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: transparent !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] svg { fill: #38bdf8 !important; }

    /* Popover & Selectbox Dropdown Menu */
    ul[role="listbox"], div[data-baseweb="popover"] *, div[data-baseweb="menu"] * {
        background-color: #0f172a !important;
        color: #ffffff !important;
    }
    li[role="option"]:hover {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }

    /* Badges */
    .badge-pill { background: rgba(56, 189, 248, 0.15); color: #38bdf8 !important; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 30px; padding: 6px 16px; font-size: 0.82rem; font-weight: 700; display: inline-flex; margin-right: 10px; margin-bottom: 10px; }
    .badge-pill-purple { background: rgba(168, 85, 247, 0.2); color: #c084fc !important; border: 1px solid rgba(168, 85, 247, 0.4); }
    .badge-pill-emerald { background: rgba(52, 211, 153, 0.2); color: #34d399 !important; border: 1px solid rgba(52, 211, 153, 0.4); }
    .badge-pill-gold { background: rgba(251, 191, 36, 0.2); color: #fbbf24 !important; border: 1px solid rgba(251, 191, 36, 0.4); }

    .official-card-container { background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(20px); border: 2px solid #38bdf8; border-radius: 24px; padding: 3.8rem 2.8rem; margin-top: 1.8rem; text-align: center; color: #ffffff !important; box-shadow: 0 25px 60px rgba(0, 0, 0, 0.65); }
    .official-badge { background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%); color: #ffffff !important; padding: 7px 24px; border-radius: 30px; font-size: 0.8rem; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1.5rem; display: inline-block; }
    .official-quote-mark { font-family: 'Playfair Display', serif; font-size: 6rem; line-height: 0.2; color: #38bdf8 !important; display: block; margin: 1.2rem auto 1.4rem auto; }
    .official-quote-text { font-size: 2.25rem; font-weight: 800; line-height: 1.5; margin: 1.5rem 0; color: #ffffff !important; }
    .official-quote-line-accent { color: #38bdf8 !important; font-weight: 800; font-size: 2.45rem; letter-spacing: 0.5px; display: block; margin: 6px 0; }
    .official-quote-line-plain { color: #ffffff !important; font-size: 2.05rem; font-weight: 700; display: block; margin: 6px 0; }
    .official-divider { width: 140px; height: 4px; background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%); margin: 2rem auto; border-radius: 4px; }
    .official-author { font-weight: 800; font-size: 1.15rem; letter-spacing: 2px; color: #38bdf8 !important; text-transform: uppercase; margin-top: 1rem; }
    .official-handle { font-size: 0.88rem; color: #93c5fd !important; margin-top: 6px; font-weight: 500; }

    /* All Buttons & Download Buttons for Dark Mode */
    .stButton > button,
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button,
    .stDownloadButton > button,
    button[data-testid="stBaseButton-secondary"],
    button[kind="secondary"],
    button[data-baseweb="button"] {
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.2rem !important;
        background: rgba(30, 41, 59, 0.9) !important;
        background-color: rgba(30, 41, 59, 0.9) !important;
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        border: 1.5px solid rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button *,
    div[data-testid="stButton"] > button *,
    div[data-testid="stDownloadButton"] > button *,
    div[data-testid="stFormSubmitButton"] > button *,
    .stDownloadButton > button *,
    button[data-testid="stBaseButton-secondary"] *,
    button[kind="secondary"] * {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    .stButton > button:hover,
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    .stDownloadButton > button:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        background-color: #0284c7 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.4) !important;
    }

    .stButton > button:hover *,
    div[data-testid="stButton"] > button:hover *,
    div[data-testid="stDownloadButton"] > button:hover *,
    div[data-testid="stFormSubmitButton"] > button:hover *,
    .stDownloadButton > button:hover *,
    button[data-testid="stBaseButton-secondary"]:hover * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stDownloadButton"] > button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 800 !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"] *,
    div[data-testid="stDownloadButton"] > button[kind="primary"] *,
    button[data-testid="stBaseButton-primary"] *,
    button[kind="primary"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .stButton > button:disabled,
    div[data-testid="stButton"] > button:disabled,
    button[disabled] {
        opacity: 0.55 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
    }

    /* Reset Stats Button Red Danger Styling for Dark Mode */
    .stButton > button[key="quiz_reset_stats_btn"],
    div[data-testid="stButton"] > button[key="quiz_reset_stats_btn"],
    button[key="quiz_reset_stats_btn"] {
        background-color: rgba(239, 68, 68, 0.2) !important;
        background: rgba(239, 68, 68, 0.2) !important;
        border: 1.5px solid #f87171 !important;
        color: #fca5a5 !important;
        -webkit-text-fill-color: #fca5a5 !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.25) !important;
    }
    .stButton > button[key="quiz_reset_stats_btn"]:hover,
    div[data-testid="stButton"] > button[key="quiz_reset_stats_btn"]:hover,
    button[key="quiz_reset_stats_btn"]:hover {
        background-color: #dc2626 !important;
        background: #dc2626 !important;
        border-color: #ef4444 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: 0 0 18px rgba(239, 68, 68, 0.5) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background: rgba(15, 23, 42, 0.8); padding: 6px; border-radius: 14px; border: 1px solid rgba(56, 189, 248, 0.25); }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 10px 20px; color: #93c5fd !important; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #0284c7 !important; border-color: #38bdf8 !important; color: #ffffff !important; font-weight: 700 !important; }

    /* Expander Header & Contrast Fixes for Dark Theme */
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1.5px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 14px !important;
        margin-bottom: 14px !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: rgba(30, 41, 59, 0.95) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stExpander"] summary * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* File Uploader Dropzone Styling for Dark Theme */
    div[data-testid="stFileUploader"],
    div[data-testid="stFileUploader"] > section,
    section[data-testid="stFileUploadDropzone"],
    div[data-baseweb="file-uploader"],
    div[data-baseweb="file-uploader"] > div {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border: 1.5px dashed rgba(56, 189, 248, 0.5) !important;
        border-radius: 14px !important;
        padding: 16px !important;
    }
    div[data-testid="stFileUploader"] *,
    section[data-testid="stFileUploadDropzone"] * {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }
    div[data-testid="stFileUploader"] button,
    section[data-testid="stFileUploadDropzone"] button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        border: 1px solid #38bdf8 !important;
        padding: 8px 18px !important;
    }
    """

    if "Light" in theme_name:
        return f"<style>{light_css}</style>"
    elif "Dark" in theme_name:
        return f"<style>{dark_css}</style>"
    else:  # System Default (Auto OS setting)
        return f"<style>{light_css}\n@media (prefers-color-scheme: dark) {{\n{dark_css}\n}}</style>"

# ---------------------------------------------------------
# Load Model & Pickles (Cached for fast offline loading)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model_path = "lstm_model.h5"
    tokenizer_path = "tokenizer.pkl"
    max_len_path = "max_len.pkl"
    dataset_path = "qoute_dataset.csv"

    model = load_model(model_path)
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        
    with open(max_len_path, "rb") as f:
        max_len = pickle.load(f)
        
    df = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else None
    
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    
    return model, tokenizer, max_len, index_to_word, df

try:
    with st.spinner("Loading offline LSTM neural model..."):
        model, tokenizer, max_len, index_to_word, dataset_df = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Error loading model files: {e}")

# ---------------------------------------------------------
# Helper Functions: Sampling, Sentiment, Audio, Card Poster Generator
# ---------------------------------------------------------
def sample_next_word(preds, temperature=1.0, sampling_mode="Temperature Sampling", top_k=50, top_p=0.9):
    preds = np.asarray(preds).astype('float64')
    if temperature <= 0.01:
        return np.argmax(preds)
        
    if "Top-K" in sampling_mode:
        top_k_val = min(max(1, int(top_k)), len(preds))
        top_k_indices = np.argsort(preds)[-top_k_val:]
        mask = np.zeros_like(preds, dtype=bool)
        mask[top_k_indices] = True
        preds[~mask] = 0.0
        if np.sum(preds) > 0:
            preds = preds / np.sum(preds)
    elif "Top-P" in sampling_mode:
        sorted_indices = np.argsort(preds)[::-1]
        sorted_preds = preds[sorted_indices]
        cumulative_preds = np.cumsum(sorted_preds)
        cutoff_idx = np.searchsorted(cumulative_preds, top_p)
        keep_indices = sorted_indices[:max(1, cutoff_idx + 1)]
        mask = np.zeros_like(preds, dtype=bool)
        mask[keep_indices] = True
        preds[~mask] = 0.0
        if np.sum(preds) > 0:
            preds = preds / np.sum(preds)
            
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text_sequence(seed_text, n_words=10, temperature=0.7, sampling_mode="Temperature Sampling", top_k=50, top_p=0.9):
    current_text = seed_text.lower().strip()
    generated_words = []
    last_top_candidates = []
    
    for step_i in range(n_words):
        seq = tokenizer.texts_to_sequences([current_text])[0]
        if not seq:
            break
        padded = pad_sequences([seq], maxlen=max_len, padding='pre')
        pred_probs = model.predict(padded, verbose=0)[0]
        
        next_idx = sample_next_word(pred_probs, temperature=temperature, sampling_mode=sampling_mode, top_k=top_k, top_p=top_p)
        next_word = index_to_word.get(next_idx, "")
        
        if not next_word or next_word == "<unk>":
            break
            
        generated_words.append(next_word)
        current_text += " " + next_word

        # Record top candidate predictions for neural insights
        top_5_indices = np.argsort(pred_probs)[-5:][::-1]
        last_top_candidates = []
        for idx in top_5_indices:
            w_cand = index_to_word.get(idx, "")
            if w_cand and w_cand != "<unk>":
                prob_pct = float(round(pred_probs[idx] * 100, 1))
                last_top_candidates.append({"word": w_cand, "prob": prob_pct})

    return " ".join(generated_words), last_top_candidates

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.35:
        mood = "✨ Highly Inspirational & Positive"
        pill_class = "badge-pill-emerald"
    elif polarity > 0.05:
        mood = "💡 Philosophical & Uplifting"
        pill_class = "badge-pill-purple"
    elif polarity < -0.05:
        mood = "🌑 Deep & Thought-Provoking"
        pill_class = "badge-pill"
    else:
        mood = "⚖️ Neutral & Balanced Thought"
        pill_class = "badge-pill-gold"
        
    normalized_score = int((polarity + 1.0) / 2.0 * 100)
    return mood, pill_class, normalized_score

def create_audio_bytes(text):
    """Generates MP3 audio buffer using English text-to-speech"""
    fp = io.BytesIO()
    tts = gTTS(text=text, lang='en', slow=False)
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def get_system_font(font_name, size):
    windir = os.environ.get("WINDIR", "C:\\Windows")
    font_path = os.path.join(windir, "Fonts", font_name)
    if os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_sparkle(draw, cx, cy, size, color):
    points = []
    for i in range(8):
        angle = i * math.pi / 4
        r = size if i % 2 == 0 else size * 0.25
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=color)

def draw_cross(draw, cx, cy, size, color, width=3):
    draw.line([(cx - size, cy), (cx + size, cy)], fill=color, width=width)
    draw.line([(cx, cy - size), (cx, cy + size)], fill=color, width=width)

DYNAMIC_POST_PALETTES = [
    {
        "name": "🌸 Soft Lavender & Indigo",
        "bg_hex": "#e0e7ff", "bg_rgb": (224, 231, 255),
        "accent_hex": "#4f46e5", "accent_rgb": (79, 70, 229),
        "dark_hex": "#1e1b4b", "dark_rgb": (30, 27, 75),
        "white_hex": "#ffffff", "white_rgb": (255, 255, 255),
        "sub_hex": "#6366f1", "sub_rgb": (99, 102, 241)
    },
    {
        "name": "💖 Dark Pink & Crisp White",
        "bg_hex": "#970747", "bg_rgb": (151, 7, 71),
        "accent_hex": "#ffffff", "accent_rgb": (255, 255, 255),
        "dark_hex": "#2b0213", "dark_rgb": (43, 2, 19),
        "white_hex": "#ffffff", "white_rgb": (255, 255, 255),
        "sub_hex": "#fbcfe8", "sub_rgb": (251, 207, 232)
    },
    {
        "name": "🩵 Powder Slate Blue",
        "bg_hex": "#afc4d1", "bg_rgb": (175, 196, 209),
        "accent_hex": "#fabe23", "accent_rgb": (250, 190, 35),
        "dark_hex": "#1a2430", "dark_rgb": (26, 36, 48),
        "white_hex": "#ffffff", "white_rgb": (255, 255, 255),
        "sub_hex": "#374b5f", "sub_rgb": (55, 75, 95)
    },
    {
        "name": "🌾 Warm Terracotta Sand",
        "bg_hex": "#f5ede2", "bg_rgb": (245, 237, 226),
        "accent_hex": "#c2410c", "accent_rgb": (194, 65, 12),
        "dark_hex": "#432818", "dark_rgb": (67, 40, 24),
        "white_hex": "#ffffff", "white_rgb": (255, 255, 255),
        "sub_hex": "#9a6b55", "sub_rgb": (154, 107, 85)
    },
    {
        "name": "🖤 Royal Charcoal & Gold",
        "bg_hex": "#0f172a", "bg_rgb": (15, 23, 42),
        "accent_hex": "#f59e0b", "accent_rgb": (245, 158, 11),
        "dark_hex": "#020617", "dark_rgb": (2, 6, 23),
        "white_hex": "#f8fafc", "white_rgb": (248, 250, 252),
        "sub_hex": "#94a3b8", "sub_rgb": (148, 163, 184)
    },
    {
        "name": "🌿 Emerald Sage & Gold",
        "bg_hex": "#2d4a3e", "bg_rgb": (45, 74, 62),
        "accent_hex": "#facc15", "accent_rgb": (250, 204, 21),
        "dark_hex": "#13231c", "dark_rgb": (19, 35, 28),
        "white_hex": "#f0fdf4", "white_rgb": (240, 253, 244),
        "sub_hex": "#86efac", "sub_rgb": (134, 239, 172)
    },
    {
        "name": "🍇 Vintage Lilac & Plum",
        "bg_hex": "#ebd6e4", "bg_rgb": (235, 214, 228),
        "accent_hex": "#9d174d", "accent_rgb": (157, 23, 77),
        "dark_hex": "#3b0764", "dark_rgb": (59, 7, 100),
        "white_hex": "#ffffff", "white_rgb": (255, 255, 255),
        "sub_hex": "#701a75", "sub_rgb": (112, 26, 117)
    },
    {
        "name": "🌅 Sunset Coral & Indigo",
        "bg_hex": "#fbcfe8", "bg_rgb": (251, 207, 232),
        "accent_hex": "#ea580c", "accent_rgb": (234, 88, 12),
        "dark_hex": "#1e1b4b", "dark_rgb": (30, 27, 75),
        "white_hex": "#ffffff", "white_rgb": (255, 255, 255),
        "sub_hex": "#4338ca", "sub_rgb": (67, 56, 202)
    },
    {
        "name": "☕ Mocha & Cream Gold",
        "bg_hex": "#3e2723", "bg_rgb": (62, 39, 35),
        "accent_hex": "#ffb74d", "accent_rgb": (255, 183, 77),
        "dark_hex": "#1b0000", "dark_rgb": (27, 0, 0),
        "white_hex": "#fff8e1", "white_rgb": (255, 248, 225),
        "sub_hex": "#d7ccc8", "sub_rgb": (215, 204, 200)
    },
    {
        "name": "🌊 Nordic Cyan & Amber",
        "bg_hex": "#155e75", "bg_rgb": (21, 94, 117),
        "accent_hex": "#fbbf24", "accent_rgb": (251, 191, 36),
        "dark_hex": "#083344", "dark_rgb": (8, 51, 68),
        "white_hex": "#ecfeff", "white_rgb": (236, 254, 255),
        "sub_hex": "#67e8f9", "sub_rgb": (103, 232, 249)
    },
    {
        "name": "💜 Lavender Mist & Deep Purple",
        "bg_hex": "#ddd6fe", "bg_rgb": (221, 214, 254),
        "accent_hex": "#d97706", "accent_rgb": (217, 119, 6),
        "dark_hex": "#4c1d95", "dark_rgb": (76, 29, 149),
        "white_hex": "#ffffff", "white_rgb": (255, 255, 255),
        "sub_hex": "#6d28d9", "sub_rgb": (109, 40, 217)
    },
    {
        "name": "🩶 Slate Gray & Mustard Gold",
        "bg_hex": "#d1d5db", "bg_rgb": (209, 213, 219),
        "accent_hex": "#d97706", "accent_rgb": (217, 119, 6),
        "dark_hex": "#111827", "dark_rgb": (17, 24, 39),
        "white_hex": "#ffffff", "white_rgb": (255, 255, 255),
        "sub_hex": "#4b5563", "sub_rgb": (75, 85, 99)
    }
]

FONT_STYLE_PRESETS = {
    "🌟 Bebas & Playfair (Poster Impact)": {
        "primary_font": "impact.ttf", "primary_size": 72,
        "secondary_font": "georgiab.ttf", "secondary_size": 54,
        "css_primary_family": "'Bebas Neue', 'Outfit', sans-serif !important",
        "css_secondary_family": "'Playfair Display', serif !important"
    },
    "✒️ Cinzel & Playfair (Royal Serif)": {
        "primary_font": "georgiab.ttf", "primary_size": 64,
        "secondary_font": "georgia.ttf", "secondary_size": 52,
        "css_primary_family": "'Cinzel', serif !important",
        "css_secondary_family": "'Playfair Display', serif !important"
    },
    "✍️ Montserrat (Modern Bold Sans)": {
        "primary_font": "arialbd.ttf", "primary_size": 64,
        "secondary_font": "segoeuib.ttf", "secondary_size": 50,
        "css_primary_family": "'Montserrat', sans-serif !important",
        "css_secondary_family": "'Montserrat', sans-serif !important"
    },
    "🎨 Dancing Script (Artistic Calligraphy)": {
        "primary_font": "segoeuii.ttf", "primary_size": 66,
        "secondary_font": "georgiai.ttf", "secondary_size": 54,
        "css_primary_family": "'Dancing Script', cursive !important",
        "css_secondary_family": "'Caveat', cursive !important"
    },
    "🌸 Pacifico (Playful Retro Script)": {
        "primary_font": "georgiai.ttf", "primary_size": 64,
        "secondary_font": "segoeuii.ttf", "secondary_size": 50,
        "css_primary_family": "'Pacifico', cursive !important",
        "css_secondary_family": "'Caveat', cursive !important"
    },
    "💻 Fira Code (Minimal Mono)": {
        "primary_font": "arial.ttf", "primary_size": 58,
        "secondary_font": "calibri.ttf", "secondary_size": 48,
        "css_primary_family": "'Fira Code', monospace !important",
        "css_secondary_family": "'Fira Code', monospace !important"
    }
}

def create_poster_card(
    quote_text,
    mood_text="INSPIRATIONAL",
    author="Khushi Singh",
    palette=None,
    font_style="🌟 Bebas & Playfair (Poster Impact)",
    text_case="UPPERCASE (Bold Poster)",
    bg_image_file=None
):
    """Generates a high-res 1080x1080 Official Designer Post Card Image buffer with solid or custom image background"""
    if palette is None:
        palette = DYNAMIC_POST_PALETTES[0]
        
    preset = FONT_STYLE_PRESETS.get(font_style, FONT_STYLE_PRESETS["🌟 Bebas & Playfair (Poster Impact)"])
    
    width, height = 1080, 1080
    bg_color = palette["bg_rgb"]
    border_outer = palette["white_rgb"]
    border_inner = palette["dark_rgb"]
    text_accent = palette["accent_rgb"]
    text_dark = palette["dark_rgb"]
    text_white = palette["white_rgb"]
    badge_bg = palette["dark_rgb"]
    badge_fg = palette["accent_rgb"]
    sub_color = palette["sub_rgb"]

    if bg_image_file is not None:
        try:
            if hasattr(bg_image_file, "seek"):
                bg_image_file.seek(0)
            user_bg = Image.open(bg_image_file).convert("RGBA")
            user_bg = user_bg.resize((width, height), Image.Resampling.LANCZOS)
            # Create dark vignette overlay for legibility
            vignette = Image.new("RGBA", (width, height), (text_dark[0], text_dark[1], text_dark[2], 190))
            img = Image.alpha_composite(user_bg, vignette).convert("RGB")
        except Exception:
            img = Image.new("RGB", (width, height), color=bg_color)
    else:
        img = Image.new("RGB", (width, height), color=bg_color)

    draw = ImageDraw.Draw(img)

    # Fonts
    font_badge = get_system_font("arialbd.ttf", 22)
    font_quote_primary = get_system_font(preset["primary_font"], preset["primary_size"])
    font_quote_secondary = get_system_font(preset["secondary_font"], preset["secondary_size"])
    font_author = get_system_font("arialbd.ttf", 26)
    font_sub = get_system_font("segoeui.ttf", 20)

    # Clean Solid Background - WHITE DOTS REMOVED!
    
    # Outer Frame & Inner Border
    draw.rectangle([(40, 40), (width - 40, height - 40)], outline=border_outer, width=4)
    draw.rectangle([(56, 56), (width - 56, height - 56)], outline=border_inner, width=2)

    # Floating Sparkles / Accents
    draw_sparkle(draw, 110, 110, 18, text_accent)
    draw_sparkle(draw, width - 110, 130, 22, text_white)
    draw_sparkle(draw, 130, height - 140, 20, text_white)
    draw_sparkle(draw, width - 120, height - 160, 18, text_accent)
    draw_cross(draw, 200, 140, 8, text_accent)
    draw_cross(draw, width - 200, height - 130, 10, text_white)

    # Top Header Pill Badge
    badge_text = mood_text.upper() if mood_text else "OFFICIAL QUOTE POST"
    bbox = font_badge.getbbox(badge_text)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bx = (width - bw) // 2
    by = 90
    draw.rectangle([(bx - 24, by - 8), (bx + bw + 24, by + bh + 14)], fill=badge_bg)
    draw.text((width // 2, by + bh//2 + 3), badge_text, fill=badge_fg, font=font_badge, anchor="mm")

    # Format Quote Lines visually
    clean_quote = quote_text.strip()
    words = clean_quote.split()
    
    if len(words) <= 6:
        lines = textwrap.wrap(clean_quote, width=15)
    elif len(words) <= 15:
        lines = textwrap.wrap(clean_quote, width=20)
    else:
        lines = textwrap.wrap(clean_quote, width=26)

    # Apply Case Formatting
    formatted_lines = []
    for l in lines:
        if "UPPERCASE" in text_case:
            formatted_lines.append(l.upper())
        elif "Title Case" in text_case:
            formatted_lines.append(l.title())
        else:
            formatted_lines.append(l.lower())

    # Vertical Centering
    line_height = 80
    total_text_height = len(formatted_lines) * line_height + 120
    start_center_y = 520 - (total_text_height // 2)

    # Large Decorative Quote Mark
    draw.text((width // 2, start_center_y), "“", fill=text_accent, font=get_system_font("georgiab.ttf", 130), anchor="mm")

    current_y = start_center_y + 80
    for i, line in enumerate(formatted_lines):
        if i % 2 == 0:
            # Primary Font
            draw.text((width // 2 + 5, current_y + 5), line, fill=text_dark, font=font_quote_primary, anchor="mm")
            for dx, dy in [(-2,0), (2,0), (0,-2), (0,2)]:
                draw.text((width // 2 + dx, current_y + dy), line, fill=text_dark, font=font_quote_primary, anchor="mm")
            draw.text((width // 2, current_y), line, fill=text_accent, font=font_quote_primary, anchor="mm")
        else:
            # Secondary Font
            draw.text((width // 2 + 3, current_y + 3), line, fill=text_dark, font=font_quote_secondary, anchor="mm")
            draw.text((width // 2, current_y), line, fill=text_white, font=font_quote_secondary, anchor="mm")
        
        current_y += line_height

    # Divider Line
    div_y = current_y + 25
    draw.line([(width//2 - 120, div_y), (width//2 + 120, div_y)], fill=text_dark, width=3)
    draw_sparkle(draw, width // 2, div_y, 9, text_accent)

    # Footer Branding with Developer Name & Subtitle
    author_clean = f"— DEVELOPER: {author.strip().upper()} —"
    subtitle_clean = "Deep Learning & Data Science Learner"
    draw.text((width // 2, height - 130), author_clean, fill=text_dark, font=font_author, anchor="mm")
    draw.text((width // 2, height - 90), subtitle_clean, fill=sub_color, font=font_sub, anchor="mm")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ---------------------------------------------------------
# Session State Setup
# ---------------------------------------------------------
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []
if "last_generated_quote" not in st.session_state:
    st.session_state["last_generated_quote"] = ""
if "last_top_candidates" not in st.session_state:
    st.session_state["last_top_candidates"] = []
if "seed_input" not in st.session_state:
    st.session_state["seed_input"] = "the world is"
if "palette_index" not in st.session_state:
    st.session_state["palette_index"] = 0

def randomize_palette():
    st.session_state["palette_index"] = int(np.random.randint(0, len(DYNAMIC_POST_PALETTES)))

def set_seed(prompt):
    st.session_state["seed_input"] = prompt
    randomize_palette()

def pick_random_seed():
    sample_seeds = [
        "the world is",
        "it is our choices",
        "there are only two ways",
        "imperfection is",
        "you can never be",
        "do what you can",
        "be yourself",
        "courage is",
        "happiness starts",
        "success begins"
    ]
    st.session_state["seed_input"] = str(np.random.choice(sample_seeds))
    randomize_palette()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 22px; padding: 10px 14px; background: rgba(255, 255, 255, 0.06); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.12); backdrop-filter: blur(10px);">
            <div style="background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%); width: 44px; height: 44px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35); flex-shrink: 0;">
                🧠
            </div>
            <div>
                <div style="font-family: 'Space Grotesk', 'Outfit', sans-serif; font-weight: 800; font-size: 1.15rem; letter-spacing: -0.5px; line-height: 1.2; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">INSPIRE AI</div>
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-top: 2px;">LSTM Neural Studio</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎨 App UI Theme Mode")
    app_theme_choice = st.selectbox(
        "Select Overall App Theme",
        [
            "💻 System Default (Auto)",
            "☀️ Light Theme (White)",
            "🌙 Dark Theme"
        ],
        index=0,
        key="global_app_theme_select",
        help="Switch between Light Theme (White), Dark Theme, or System Default (automatic OS preference)."
    )
    
    # Inject active theme CSS
    st.markdown(get_theme_css(app_theme_choice), unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🧠 Model Architecture")
    st.markdown("""
    - **Network:** Deep LSTM Layer
    - **Embedding Dim:** 50
    - **LSTM Units:** 128
    - **Vocab Size:** 8,978 Words
    - **Max Sequence:** 745 Tokens
    """)
    st.divider()
    
    st.markdown("### ⚙️ Generation Controls")
    creativity = st.slider(
        "Creativity (Temperature)",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Higher values make output more varied; lower values make it more predictable."
    )
    
    num_words = st.slider(
        "Words to Generate",
        min_value=5,
        max_value=30,
        value=12,
        step=1
    )

    sampling_mode_choice = st.selectbox(
        "Sampling Strategy Mode",
        [
            "🌡️ Temperature Sampling",
            "🎯 Top-K Sampling (Truncated)",
            "🌊 Top-P Nucleus Sampling"
        ],
        index=0,
        help="Select how the next word probabilities are sampled from the model."
    )

    top_k_param = 50
    top_p_param = 0.9

    if "Top-K" in sampling_mode_choice:
        top_k_param = st.slider("Top-K Candidates", min_value=1, max_value=100, value=50, step=1)
    elif "Top-P" in sampling_mode_choice:
        top_p_param = st.slider("Top-P Cumulative Mass", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    
    st.divider()
    st.markdown("### 👤 Developer Info")
    dev_author_input = st.text_input(
        "Developer Name",
        value="Khushi Singh",
        help="Developer name displayed on poster cards and e-books"
    )

    # Sidebar Dataset Quick Search Engine
    st.divider()
    st.markdown("### 🔍 Live Dataset Quick Search")
    side_search = st.text_input("Search 3,000+ Quotes...", key="side_search_query", placeholder="e.g. 'courage' or 'Einstein'")
    if side_search.strip() and dataset_df is not None:
        q_col = 'quote' if 'quote' in dataset_df.columns else dataset_df.columns[0]
        matches = dataset_df[dataset_df[q_col].astype(str).str.contains(side_search, case=False, na=False)]
        if not matches.empty:
            st.caption(f"Found {len(matches)} matching quotes:")
            for idx_m, row_m in matches.head(3).iterrows():
                q_text_m = str(row_m[q_col])
                words_m = q_text_m.split()[:4]
                seed_m = " ".join(words_m)
                if st.button(f"🌱 Seed: '{seed_m}...'", key=f"side_seed_btn_{idx_m}"):
                    set_seed(seed_m)
                    st.rerun()
        else:
            st.caption("No matching quotes found.")



# ---------------------------------------------------------
# Main App Header & Navigation
# ---------------------------------------------------------
st.markdown('<span class="badge-pill badge-pill-purple" style="font-size: 0.78rem; font-weight: 800; letter-spacing: 1.5px; padding: 4px 14px; text-transform: uppercase; margin-bottom: 12px;">✨ DEEP LEARNING LSTM PLATFORM v3.0 ULTIMATE</span>', unsafe_allow_html=True)

st.markdown('<div class="main-title">InspireAI — Neural Quote Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Page AI Studio: Neural Composition, Dedicated Poster Studio, AI vs Human Battle, Poster Exhibition, Architecture Lab & E-Book Creator</div>', unsafe_allow_html=True)

main_nav_tab1, main_nav_tab2, main_nav_tab3, main_nav_tab4, main_nav_tab5, main_nav_tab6, main_nav_tab7 = st.tabs([
    "✨ AI Studio",
    "🎨 Poster Studio",
    "🎮 AI vs Human Quiz",
    "🏛️ Poster Gallery",
    "🎓 DL Model Lab",
    "📖 Quote Book Creator",
    "📊 Analytics & Dataset"
])

# ---------------------------------------------------------
# TAB 1: AI STUDIO
# ---------------------------------------------------------
with main_nav_tab1:
    # Prompt Input Section
    st.markdown("##### 💡 Enter a Seed Prompt or Pick a Quick Suggestion:")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.button("✨ The world is", on_click=set_seed, args=("the world is",), use_container_width=True, key="studio_btn1")
    with col2:
        st.button("🌱 Life is a", on_click=set_seed, args=("life is a",), use_container_width=True, key="studio_btn2")
    with col3:
        st.button("🔥 Courage means", on_click=set_seed, args=("courage means",), use_container_width=True, key="studio_btn3")
    with col4:
        st.button("💡 Happiness starts", on_click=set_seed, args=("happiness starts",), use_container_width=True, key="studio_btn4")
    with col5:
        st.button("🚀 Success is", on_click=set_seed, args=("success is",), use_container_width=True, key="studio_btn5")

    user_seed = st.text_input(
        "Seed Text Input",
        key="seed_input",
        placeholder="Type a starting phrase... (e.g. 'there are only two ways')",
        label_visibility="collapsed"
    )

    col_gen, col_rand = st.columns([3, 1])

    with col_gen:
        generate_btn = st.button("✨ Generate Quote", type="primary", use_container_width=True, key="studio_gen_btn")
    with col_rand:
        st.button("🎲 Random Seed", on_click=pick_random_seed, use_container_width=True, key="studio_rand_btn")

    if generate_btn:
        if not user_seed.strip():
            st.warning("Please enter a seed phrase first!")
        elif assets_loaded:
            with st.spinner("✨ LSTM Model is composing your quote..."):
                generated_suffix, top_cands = generate_text_sequence(
                    user_seed,
                    n_words=num_words,
                    temperature=creativity,
                    sampling_mode=sampling_mode_choice,
                    top_k=top_k_param,
                    top_p=top_p_param
                )
                full_quote = f"{user_seed.strip()} {generated_suffix}".strip()
                
                if len(full_quote) > 0:
                    full_quote = full_quote[0].upper() + full_quote[1:] + "."
                    
                st.session_state["last_generated_quote"] = full_quote
                st.session_state["last_top_candidates"] = top_cands
                randomize_palette()

    if st.session_state["last_generated_quote"]:
        quote_content = st.session_state["last_generated_quote"]
        
        mood_title, mood_class, pos_score = analyze_sentiment(quote_content)
        
        active_p = DYNAMIC_POST_PALETTES[st.session_state["palette_index"] % len(DYNAMIC_POST_PALETTES)].copy()
        active_font_preset = FONT_STYLE_PRESETS["🌟 Bebas & Playfair (Poster Impact)"]

        clean_q = quote_content.strip()
        words_list = clean_q.split()
        if len(words_list) <= 6:
            q_lines = textwrap.wrap(clean_q, width=15)
        elif len(words_list) <= 15:
            q_lines = textwrap.wrap(clean_q, width=20)
        else:
            q_lines = textwrap.wrap(clean_q, width=26)

        quote_html_chunks = []
        for idx_l, line_str in enumerate(q_lines):
            c_line = line_str.upper()

            if idx_l % 2 == 0:
                quote_html_chunks.append(
                    f'<span class="official-quote-line-accent" style="font-family: {active_font_preset["css_primary_family"]}; color: {active_p["accent_hex"]}; text-shadow: 3px 3px 0px {active_p["dark_hex"]}, -1px -1px 0px {active_p["dark_hex"]}, 1px -1px 0px {active_p["dark_hex"]}, -1px 1px 0px {active_p["dark_hex"]};">{c_line}</span>'
                )
            else:
                quote_html_chunks.append(
                    f'<span class="official-quote-line-plain" style="font-family: {active_font_preset["css_secondary_family"]}; color: {active_p["white_hex"]};">{c_line}</span>'
                )

        formatted_quote_html = "".join(quote_html_chunks)

        st.markdown(f"""
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800;900&family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,400&family=Cinzel:wght@700;900&family=Montserrat:wght@700;800;900&family=Bebas+Neue&family=Dancing+Script:wght@700&family=Caveat:wght@700&family=Pacifico&family=Fira+Code:wght@700&display=swap">

        <div class="official-card-container" style="background-color: {active_p['bg_hex']}; border-color: {active_p['white_hex']}; outline-color: {active_p['dark_hex']}; color: {active_p['dark_hex']};">
            <div style="margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <span class="{mood_class}">Mood: {mood_title}</span>
                <span class="official-badge" style="background-color: {active_p['dark_hex']}; color: {active_p['accent_hex']};">✦ OFFICIAL QUOTE POST ✦</span>
            </div>
            <span class="official-quote-mark" style="color: {active_p['accent_hex']};">“</span>
            <div class="official-quote-text">
                {formatted_quote_html}
            </div>
            <div class="official-divider" style="background-color: {active_p['dark_hex']};"></div>
            <div class="official-author" style="color: {active_p['dark_hex']};">— DEVELOPER: {dev_author_input.strip().upper()} —</div>
            <div class="official-handle" style="color: {active_p['sub_hex']};">Deep Learning & Data Science Learner</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        c_fav, c_listen, c_dl, c_cust = st.columns([1.8, 2.2, 2.2, 2.8])
        
        with c_fav:
            if st.button("❤️ Save Favorite", use_container_width=True, key="studio_save_fav_btn"):
                if quote_content not in st.session_state["favorites"]:
                    st.session_state["favorites"].append(quote_content)
                    st.toast("Saved to your favorites list!", icon="❤️")

        with c_listen:
            play_audio_clicked = st.button("🔊 Listen Audio", use_container_width=True, type="secondary", key="studio_listen_btn")
                
        with c_dl:
            try:
                poster_buf = create_poster_card(
                    quote_content,
                    mood_text=mood_title,
                    author=dev_author_input,
                    palette=active_p,
                )
                st.download_button(
                    label="🖼️ Quick PNG",
                    data=poster_buf,
                    file_name="inspire_ai_official_quote_card.png",
                    mime="image/png",
                    use_container_width=True,
                    key="studio_download_poster_btn"
                )
            except Exception as err:
                st.caption(f"Poster error: {err}")

        try:
            audio_buf = create_audio_bytes(quote_content)
            st.audio(audio_buf, format="audio/mp3", autoplay=play_audio_clicked)
        except Exception as err:
            st.caption("🔊 Audio preview unavailable offline")

        if "last_top_candidates" in st.session_state and st.session_state["last_top_candidates"]:
            st.write("")
            with st.expander("🧠 Deep Learning Neural Token Probability Insights (Live Softmax)", expanded=False):
                st.caption("Top candidate next-words evaluated by the LSTM Neural Network at the final prediction step:")
                for cand in st.session_state["last_top_candidates"]:
                    col_w, col_p = st.columns([1.2, 4])
                    with col_w:
                        st.markdown(f"**'{cand['word']}'**")
                    with col_p:
                        st.progress(min(1.0, cand["prob"] / 100.0), text=f"{cand['prob']}% probability")

        st.write("")
        with st.expander("📲 One-Click Social Media Formatting & Captions", expanded=False):
            st.markdown("##### 📸 Instagram Post Caption")
            insta_text = f"“{quote_content}”\n\n— Developer: {dev_author_input.strip().title()}\n\n#DeepLearning #InspireAI #Quotes #MachineLearning #Python #Mindset #DailyMotivation"
            st.code(insta_text, language="markdown")
            
            col_tw, col_li = st.columns(2)
            with col_tw:
                st.markdown("##### 🐦 Twitter / X Snippet")
                tw_text = f"“{quote_content}” — @{dev_author_input.strip().replace(' ', '')} #InspireAI #DeepLearning"
                st.code(tw_text, language="markdown")
            with col_li:
                st.markdown("##### 💼 LinkedIn Professional Snippet")
                li_text = f"Sharing an AI-generated quote composed by my Deep Learning LSTM model:\n\n“{quote_content}”\n\nBuilt with TensorFlow & Keras. #ArtificialIntelligence #MachineLearning #NLP"
                st.code(li_text, language="markdown")

    st.divider()
    st.markdown("### ❤️ Saved Favorites List")
    if len(st.session_state["favorites"]) == 0:
        st.info("No saved quotes yet. Click '❤️ Save to Favorites' above to bookmark generated quotes!")
    else:
        for idx, fav in enumerate(st.session_state["favorites"]):
            st.markdown(f"**{idx+1}.** *“{fav}”*")

# ---------------------------------------------------------
# TAB 2: DEDICATED POSTER CANVAS STUDIO
# ---------------------------------------------------------
with main_nav_tab2:
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1.5px solid rgba(99, 102, 241, 0.3); border-radius: 20px; padding: 20px 24px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #ffffff;">🎨 Dedicated HD Poster Design Studio</div>
            <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 4px;">Customize quote typography, color palettes, background wallpapers & export print-ready 1080x1080 PNG posters.</div>
        </div>
        <div style="display: flex; gap: 10px;">
            <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid #6366f1; color: #818cf8; padding: 6px 14px; border-radius: 30px; font-size: 0.8rem; font-weight: 800;">📐 1080x1080 HD Canvas</span>
            <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #34d399; padding: 6px 14px; border-radius: 30px; font-size: 0.8rem; font-weight: 800;">⚡ Instant Live Render</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    studio_c1, studio_c2 = st.columns([1.15, 1.1])
    
    with studio_c1:
        st.markdown("#### 🛠️ Studio Control Panel")
        
        # GROUP 1: Content & Text Branding
        with st.expander("📝 1. Content & Text Branding", expanded=True):
            default_q_text = st.session_state.get("last_generated_quote", "The world is a canvas of infinite possibilities.")
            poster_quote_input = st.text_area(
                "Quote Text to Render",
                value=default_q_text,
                height=95,
                key="studio_poster_quote_input",
                help="Edit or type the quote text to display on your poster."
            )
            
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                poster_badge_input = st.text_input(
                    "Header Pill Badge",
                    value="OFFICIAL QUOTE POST",
                    key="studio_poster_badge_input"
                )
            with c_b2:
                text_case_choice = st.selectbox(
                    "Capitalization",
                    ["UPPERCASE (Bold Poster)", "Title Case (Classic)", "lowercase (minimal)"],
                    index=0,
                    key="card_text_case_select"
                )

        # GROUP 2: Color Palette & Custom Accents
        with st.expander("🎨 2. Color Palette & Custom Themes", expanded=True):
            palette_options = ["🎲 Random (Dynamic Color per Quote)"] + [p["name"] for p in DYNAMIC_POST_PALETTES]
            
            col_p1, col_p2 = st.columns([2.5, 1.5])
            with col_p1:
                card_theme_choice = st.selectbox(
                    "Theme Palette Preset",
                    palette_options,
                    index=0,
                    key="card_theme_choice_select"
                )
            with col_p2:
                st.write("")
                st.write("")
                if st.button("🎲 Shuffle", use_container_width=True, key="studio_shuffle_palette_btn"):
                    randomize_palette()
                    st.rerun()

            col_acc1, col_acc2 = st.columns([1, 2])
            with col_acc1:
                custom_accent_hex = st.color_picker(
                    "Custom Accent Color",
                    value="#FABE23",
                    key="custom_color_picker_input"
                )
            with col_acc2:
                st.write("")
                st.write("")
                use_custom_accent = st.checkbox("Apply Custom Color Overlay", value=False, key="use_custom_accent_chk")

        # GROUP 3: Typography & Background Image
        with st.expander("🔤 3. Typography & Background Wallpaper", expanded=True):
            font_style_choice = st.selectbox(
                "Typography Font Pairing",
                list(FONT_STYLE_PRESETS.keys()),
                index=0,
                key="card_font_style_select"
            )
            
            custom_bg_wallpaper = st.file_uploader(
                "Upload Background Wallpaper Image (Optional)",
                type=["png", "jpg", "jpeg"],
                key="poster_bg_file_uploader",
                help="Upload an image to blend into the poster background with vignette overlay."
            )

    with studio_c2:
        st.markdown("#### 🖼️ Live Canvas Preview & Export")
        
        if card_theme_choice == "🎲 Random (Dynamic Color per Quote)":
            active_p_studio = DYNAMIC_POST_PALETTES[st.session_state["palette_index"] % len(DYNAMIC_POST_PALETTES)].copy()
        else:
            active_p_studio = next((p for p in DYNAMIC_POST_PALETTES if p["name"] == card_theme_choice), DYNAMIC_POST_PALETTES[0]).copy()

        if use_custom_accent:
            active_p_studio["accent_hex"] = custom_accent_hex
            h_hex = custom_accent_hex.lstrip("#")
            active_p_studio["accent_rgb"] = tuple(int(h_hex[i:i+2], 16) for i in (0, 2, 4))

        if poster_quote_input.strip():
            try:
                poster_buf_studio = create_poster_card(
                    poster_quote_input,
                    mood_text=poster_badge_input,
                    author=dev_author_input,
                    palette=active_p_studio,
                    font_style=font_style_choice,
                    text_case=text_case_choice,
                    bg_image_file=custom_bg_wallpaper
                )
                
                st.image(poster_buf_studio, use_container_width=True)
                
                st.download_button(
                    label="🖼️ Download High-Resolution Poster PNG (1080x1080)",
                    data=poster_buf_studio,
                    file_name="inspire_ai_custom_poster.png",
                    mime="image/png",
                    use_container_width=True,
                    key="studio_canvas_download_btn"
                )
            except Exception as err:
                st.error(f"Canvas render error: {err}")
        else:
            st.info("💡 Enter or generate a quote to preview your live poster card!")

# ---------------------------------------------------------
# TAB 3: AI VS HUMAN QUIZ GAME (Gamified Turing Challenge)
# ---------------------------------------------------------
with main_nav_tab3:
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "quiz_total" not in st.session_state:
        st.session_state["quiz_total"] = 0
    if "quiz_streak" not in st.session_state:
        st.session_state["quiz_streak"] = 0
    if "quiz_best_streak" not in st.session_state:
        st.session_state["quiz_best_streak"] = 0
    if "quiz_difficulty" not in st.session_state:
        st.session_state["quiz_difficulty"] = "Medium (Balanced)"
    if "quiz_current" not in st.session_state:
        st.session_state["quiz_current"] = None

    # Calculate Gamified Accuracy Rank & Metrics
    total_q = st.session_state["quiz_total"]
    wins_q = st.session_state["quiz_score"]
    losses_q = total_q - wins_q
    acc_pct = int((wins_q / total_q) * 100) if total_q > 0 else 0
    
    if total_q == 0:
        rank_badge = "Unranked (Play First Round)"
    elif acc_pct >= 90:
        rank_badge = "AI Whisperer Legend"
    elif acc_pct >= 75:
        rank_badge = "Turing Master"
    elif acc_pct >= 50:
        rank_badge = "AI Detective"
    else:
        rank_badge = "AI Novice"

    # Glassmorphic Header Card
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1.5px solid rgba(99, 102, 241, 0.3); border-radius: 20px; padding: 20px 24px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff;">AI vs. Human Quote Turing Challenge</div>
            <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 4px;">Test your intuition! Can you spot which quote was generated by the <b>LSTM Neural Model</b> and which by a <b>Real Author</b>?</div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid #6366f1; color: #818cf8; padding: 6px 14px; border-radius: 30px; font-size: 0.82rem; font-weight: 800;">{rank_badge}</span>
            <span style="background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; color: #fbbf24; padding: 6px 14px; border-radius: 30px; font-size: 0.82rem; font-weight: 800;">Best Streak: {st.session_state['quiz_best_streak']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Compact Real KPI Cards Grid
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin-bottom: 18px;">
        <div style="background: rgba(30, 41, 59, 0.6); border: 1.5px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 12px 8px; text-align: center;">
            <div style="font-size: 0.72rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Total Rounds</div>
            <div style="font-size: 1.55rem; font-weight: 900; color: #ffffff; margin-top: 4px;">{total_q}</div>
        </div>
        <div style="background: rgba(16, 185, 129, 0.08); border: 1.5px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 12px 8px; text-align: center;">
            <div style="font-size: 0.72rem; font-weight: 800; color: #34d399; text-transform: uppercase; letter-spacing: 0.5px;">Wins (Won)</div>
            <div style="font-size: 1.55rem; font-weight: 900; color: #34d399; margin-top: 4px;">{wins_q}</div>
        </div>
        <div style="background: rgba(239, 68, 68, 0.08); border: 1.5px solid rgba(239, 68, 68, 0.35); border-radius: 12px; padding: 12px 8px; text-align: center;">
            <div style="font-size: 0.72rem; font-weight: 800; color: #f87171; text-transform: uppercase; letter-spacing: 0.5px;">Losses (Lost)</div>
            <div style="font-size: 1.55rem; font-weight: 900; color: #f87171; margin-top: 4px;">{losses_q}</div>
        </div>
        <div style="background: rgba(56, 189, 248, 0.08); border: 1.5px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 12px 8px; text-align: center;">
            <div style="font-size: 0.72rem; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px;">Accuracy Rate</div>
            <div style="font-size: 1.55rem; font-weight: 900; color: #38bdf8; margin-top: 4px;">{acc_pct}%</div>
        </div>
        <div style="background: rgba(245, 158, 11, 0.08); border: 1.5px solid rgba(245, 158, 11, 0.35); border-radius: 12px; padding: 12px 8px; text-align: center;">
            <div style="font-size: 0.72rem; font-weight: 800; color: #fbbf24; text-transform: uppercase; letter-spacing: 0.5px;">Active Streak</div>
            <div style="font-size: 1.55rem; font-weight: 900; color: #fbbf24; margin-top: 4px;">{st.session_state['quiz_streak']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Proper Interactive Buttons for Difficulty Mode
    st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #94a3b8; margin-bottom: 8px;'>Difficulty Mode:</div>", unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3, c_btn_rst = st.columns([1.5, 1.5, 1.8, 1.4])
    
    cur_diff = st.session_state["quiz_difficulty"]
    
    with c_btn1:
        if st.button("Easy (Varied AI)", key="diff_btn_easy", use_container_width=True, type="primary" if cur_diff == "Easy (Varied AI)" else "secondary"):
            st.session_state["quiz_difficulty"] = "Easy (Varied AI)"
            st.rerun()
    with c_btn2:
        if st.button("Medium (Balanced)", key="diff_btn_med", use_container_width=True, type="primary" if cur_diff == "Medium (Balanced)" else "secondary"):
            st.session_state["quiz_difficulty"] = "Medium (Balanced)"
            st.rerun()
    with c_btn3:
        if st.button("Hard (Turing Realistic)", key="diff_btn_hard", use_container_width=True, type="primary" if cur_diff == "Hard (Turing Realistic)" else "secondary"):
            st.session_state["quiz_difficulty"] = "Hard (Turing Realistic)"
            st.rerun()
    with c_btn_rst:
        if st.button("Reset Stats", key="quiz_reset_stats_btn", use_container_width=True):
            st.session_state["quiz_score"] = 0
            st.session_state["quiz_total"] = 0
            st.session_state["quiz_streak"] = 0
            st.toast("Game statistics reset!")
            st.rerun()

    st.divider()

    def load_new_quiz_question():
        if dataset_df is not None:
            q_col = 'quote' if 'quote' in dataset_df.columns else dataset_df.columns[0]
            human_row = dataset_df.sample(1).iloc[0]
            h_quote = str(human_row[q_col]).strip()
            h_author = str(human_row['Author']).strip() if 'Author' in dataset_df.columns else "Famous Author"
        else:
            h_quote = "The only limit to our realization of tomorrow will be our doubts of today."
            h_author = "Franklin D. Roosevelt"
            
        sample_seeds = ["courage is", "life is a", "the world is", "happiness starts", "success begins", "there are only", "imperfection is", "be yourself"]
        rand_seed = str(np.random.choice(sample_seeds))
        
        quiz_diff_choice = st.session_state["quiz_difficulty"]
        temp_val = 0.9 if "Easy" in quiz_diff_choice else (0.7 if "Medium" in quiz_diff_choice else 0.45)
        ai_suffix, top_cands_q = generate_text_sequence(rand_seed, n_words=11, temperature=temp_val)
        ai_quote = f"{rand_seed} {ai_suffix}".strip()
        if len(ai_quote) > 0:
            ai_quote = ai_quote[0].upper() + ai_quote[1:] + "."
            
        mood_title, mood_class, pos_score = analyze_sentiment(ai_quote)
            
        ai_first = bool(np.random.choice([True, False]))
        st.session_state["quiz_current"] = {
            "h_quote": h_quote,
            "h_author": h_author,
            "ai_quote": ai_quote,
            "ai_first": ai_first,
            "top_candidates": top_cands_q,
            "mood_title": mood_title,
            "pos_score": pos_score,
            "temp_val": temp_val,
            "answered": False,
            "user_correct": False
        }

    if st.session_state["quiz_current"] is None:
        load_new_quiz_question()

    q_data = st.session_state["quiz_current"]
    
    if q_data["ai_first"]:
        option_A_text, option_A_type = q_data["ai_quote"], "AI"
        option_B_text, option_B_type = q_data["h_quote"], "Human"
    else:
        option_A_text, option_A_type = q_data["h_quote"], "Human"
        option_B_text, option_B_type = q_data["ai_quote"], "AI"

    # Symmetrical Cards Layout for Option A vs Option B
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.08); padding: 24px; border-radius: 20px; border: 1.5px solid rgba(99, 102, 241, 0.35); text-align: center; min-height: 200px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid #6366f1; color: #818cf8; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 800; letter-spacing: 2px;">OPTION A</span>
                <div style="font-size: 1.2rem; font-weight: 700; margin-top: 16px; line-height: 1.55;">“{option_A_text}”</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        c_pick_a, c_audio_a = st.columns([2.5, 1.5])
        with c_pick_a:
            if st.button("Pick Option A as AI", key="quiz_pick_a", use_container_width=True, disabled=q_data["answered"], type="primary"):
                if not q_data["answered"]:
                    q_data["answered"] = True
                    st.session_state["quiz_total"] += 1
                    if option_A_type == "AI":
                        q_data["user_correct"] = True
                        st.session_state["quiz_score"] += 1
                        st.session_state["quiz_streak"] += 1
                        if st.session_state["quiz_streak"] > st.session_state["quiz_best_streak"]:
                            st.session_state["quiz_best_streak"] = st.session_state["quiz_streak"]
                    else:
                        st.session_state["quiz_streak"] = 0
                    st.rerun()
        with c_audio_a:
            if st.button("Listen A", key="quiz_listen_a", use_container_width=True):
                try:
                    aud_a = create_audio_bytes(option_A_text)
                    st.audio(aud_a, format="audio/mp3", autoplay=True)
                except Exception:
                    st.toast("Audio preview unavailable")

    with col_q2:
        st.markdown(f"""
        <div style="background: rgba(236, 72, 153, 0.08); padding: 24px; border-radius: 20px; border: 1.5px solid rgba(236, 72, 153, 0.35); text-align: center; min-height: 200px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <span style="background: rgba(236, 72, 153, 0.2); border: 1px solid #ec4899; color: #f472b6; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 800; letter-spacing: 2px;">OPTION B</span>
                <div style="font-size: 1.2rem; font-weight: 700; margin-top: 16px; line-height: 1.55;">“{option_B_text}”</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        c_pick_b, c_audio_b = st.columns([2.5, 1.5])
        with c_pick_b:
            if st.button("Pick Option B as AI", key="quiz_pick_b", use_container_width=True, disabled=q_data["answered"], type="primary"):
                if not q_data["answered"]:
                    q_data["answered"] = True
                    st.session_state["quiz_total"] += 1
                    if option_B_type == "AI":
                        q_data["user_correct"] = True
                        st.session_state["quiz_score"] += 1
                        st.session_state["quiz_streak"] += 1
                        if st.session_state["quiz_streak"] > st.session_state["quiz_best_streak"]:
                            st.session_state["quiz_best_streak"] = st.session_state["quiz_streak"]
                    else:
                        st.session_state["quiz_streak"] = 0
                    st.rerun()
        with c_audio_b:
            if st.button("Listen B", key="quiz_listen_b", use_container_width=True):
                try:
                    aud_b = create_audio_bytes(option_B_text)
                    st.audio(aud_b, format="audio/mp3", autoplay=True)
                except Exception:
                    st.toast("Audio preview unavailable")

    # Post-Answer Outcome & Insights Reveal
    if q_data["answered"]:
        st.write("")
        ai_opt_letter = 'A' if option_A_type == 'AI' else 'B'
        human_opt_letter = 'B' if option_A_type == 'AI' else 'A'
        
        if q_data["user_correct"]:
            st.success(f"Spot On! Correct Choice! Option {ai_opt_letter} was generated by the Deep Learning LSTM Model! Option {human_opt_letter} was written by {q_data['h_author']}. Active Streak: {st.session_state['quiz_streak']}!")
            st.balloons()
        else:
            st.error(f"Ooh, tricky! Option {ai_opt_letter} was actually generated by the Neural AI Model! Option {human_opt_letter} was written by {q_data['h_author']}.")

        with st.expander("Neural Model & Sentiment Insights for this Round", expanded=True):
            c_inf1, c_inf2, c_inf3 = st.columns(3)
            with c_inf1:
                st.markdown(f"**AI Quote Mood:**\n{q_data['mood_title']}")
            with c_inf2:
                st.markdown(f"**Temperature:**\n`{q_data['temp_val']}` ({st.session_state['quiz_difficulty'].split(' ')[0]})")
            with c_inf3:
                st.markdown(f"**Real Author:**\n{q_data['h_author']}")
                
            if q_data.get("top_candidates"):
                st.write("")
                st.caption("Top Softmax next-word probabilities evaluated by LSTM for the AI quote:")
                for cand in q_data["top_candidates"]:
                    c_w, c_p = st.columns([1.2, 4])
                    with c_w:
                        st.markdown(f"**'{cand['word']}'**")
                    with c_p:
                        st.progress(min(1.0, cand["prob"] / 100.0), text=f"{cand['prob']}% probability")
            
        st.write("")
        if st.button("Play Next Challenge Round", key="quiz_next_round_btn", type="primary", use_container_width=True):
            load_new_quiz_question()
            st.rerun()

# ---------------------------------------------------------
# TAB 4: POSTER GALLERY EXHIBITION
# ---------------------------------------------------------
with main_nav_tab4:
    st.markdown("### 🏛️ Neural Poster Exhibition & Inspiration Wall")
    st.markdown("Browse 12 curated high-resolution poster designs with unique typography styles, color palettes & borders:")
    
    gallery_samples = [
        {
            "quote": "The world is a canvas of infinite possibilities.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[0],
            "font_style": "🌟 Bebas & Playfair (Poster Impact)",
            "design_name": "✨ Soft Indigo Glass",
            "bg_css": "background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%) !important; background-color: #e0e7ff !important;",
            "border_css": "border: 2px solid #6366f1; box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);",
            "quote_color": "#1e1b4b",
            "author_color": "#4338ca",
            "badge_bg": "#1e1b4b",
            "badge_fg": "#ffffff"
        },
        {
            "quote": "Courage is not the absence of fear but the triumph over it.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[1],
            "font_style": "🌸 Pacifico (Playful Retro Script)",
            "design_name": "💖 Crimson Velvet",
            "bg_css": "background: linear-gradient(135deg, #970747 0%, #5a042a 100%) !important; background-color: #970747 !important;",
            "border_css": "border: 3px solid #ffffff; outline: 2px solid #2b0213; box-shadow: 0 15px 35px rgba(151, 7, 71, 0.5);",
            "quote_color": "#ffffff",
            "author_color": "#fbcfe8",
            "badge_bg": "#2b0213",
            "badge_fg": "#ffffff"
        },
        {
            "quote": "Success begins with the decision to try.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[2],
            "font_style": "✍️ Montserrat (Modern Bold Sans)",
            "design_name": "🩵 Slate Minimalist",
            "bg_css": "background: linear-gradient(135deg, #afc4d1 0%, #90a4ae 100%) !important; background-color: #afc4d1 !important;",
            "border_css": "border: 2px solid #fabe23; box-shadow: 0 10px 25px rgba(26, 36, 48, 0.3);",
            "quote_color": "#0f172a",
            "author_color": "#1e293b",
            "badge_bg": "#0f172a",
            "badge_fg": "#fabe23"
        },
        {
            "quote": "Do what you can with all you have wherever you are.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[3],
            "font_style": "🎨 Dancing Script (Artistic Calligraphy)",
            "design_name": "🌾 Terracotta Sand",
            "bg_css": "background: linear-gradient(135deg, #f5ede2 0%, #e6ccb2 100%) !important; background-color: #f5ede2 !important;",
            "border_css": "border: 2px dashed #c2410c; box-shadow: 0 10px 25px rgba(194, 65, 12, 0.25);",
            "quote_color": "#432818",
            "author_color": "#c2410c",
            "badge_bg": "#432818",
            "badge_fg": "#ffffff"
        },
        {
            "quote": "Imperfection is beauty and madness is genius.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[4],
            "font_style": "✒️ Cinzel & Playfair (Royal Serif)",
            "design_name": "🏆 Royal Obsidian Gold",
            "bg_css": "background: linear-gradient(135deg, #18181b 0%, #09090b 100%) !important; background-color: #18181b !important;",
            "border_css": "border: 2px solid #f59e0b; box-shadow: 0 0 25px rgba(245, 158, 11, 0.4);",
            "quote_color": "#fbbf24",
            "author_color": "#ffffff",
            "badge_bg": "#27272a",
            "badge_fg": "#fbbf24"
        },
        {
            "quote": "Happiness starts from within your own thoughts.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[5],
            "font_style": "✒️ Cinzel & Playfair (Royal Serif)",
            "design_name": "🌿 Emerald Forest & Gold",
            "bg_css": "background: linear-gradient(135deg, #064e3b 0%, #022c22 100%) !important; background-color: #064e3b !important;",
            "border_css": "border: 2px solid #34d399; box-shadow: 0 12px 30px rgba(6, 78, 59, 0.4);",
            "quote_color": "#a7f3d0",
            "author_color": "#ffffff",
            "badge_bg": "#13231c",
            "badge_fg": "#34d399"
        },
        {
            "quote": "Life is a journey of continuous discovery.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[6],
            "font_style": "🌸 Pacifico (Playful Retro Script)",
            "design_name": "🍇 Vintage Lilac & Plum",
            "bg_css": "background: linear-gradient(135deg, #ebd6e4 0%, #d8b4e2 100%) !important; background-color: #ebd6e4 !important;",
            "border_css": "border: 2px solid #9d174d; box-shadow: 0 10px 25px rgba(157, 23, 77, 0.3);",
            "quote_color": "#3b0764",
            "author_color": "#9d174d",
            "badge_bg": "#3b0764",
            "badge_fg": "#ffffff"
        },
        {
            "quote": "Be yourself, everyone else is already taken.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[7],
            "font_style": "🎨 Dancing Script (Artistic Calligraphy)",
            "design_name": "🌅 Sunset Coral Glow",
            "bg_css": "background: linear-gradient(135deg, #2e1065 0%, #1e1b4b 100%) !important; background-color: #2e1065 !important;",
            "border_css": "border: 2px solid #ea580c; box-shadow: 0 12px 30px rgba(234, 88, 12, 0.4);",
            "quote_color": "#ff8855",
            "author_color": "#ffffff",
            "badge_bg": "#4338ca",
            "badge_fg": "#ff8855"
        },
        {
            "quote": "There are only two ways to live your life.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[8],
            "font_style": "✍️ Montserrat (Modern Bold Sans)",
            "design_name": "☕ Mocha Cream & Gold",
            "bg_css": "background: linear-gradient(135deg, #3e2723 0%, #1b0000 100%) !important; background-color: #3e2723 !important;",
            "border_css": "border: 2px solid #ffb74d; box-shadow: 0 10px 25px rgba(62, 39, 35, 0.4);",
            "quote_color": "#ffb74d",
            "author_color": "#ffffff",
            "badge_bg": "#270000",
            "badge_fg": "#ffb74d"
        },
        {
            "quote": "You can never be overdressed or overeducated.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[9],
            "font_style": "💻 Fira Code (Minimal Mono)",
            "design_name": "⚡ Cyber Cyan Monospace",
            "bg_css": "background: linear-gradient(135deg, #083344 0%, #020617 100%) !important; background-color: #083344 !important;",
            "border_css": "border: 2px solid #67e8f9; box-shadow: 0 0 25px rgba(103, 232, 249, 0.45);",
            "quote_color": "#67e8f9",
            "author_color": "#ffffff",
            "badge_bg": "#0f172a",
            "badge_fg": "#67e8f9"
        },
        {
            "quote": "It is our choices that show what we truly are.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[10],
            "font_style": "🌟 Bebas & Playfair (Poster Impact)",
            "design_name": "💜 Lavender Mist Royal",
            "bg_css": "background: linear-gradient(135deg, #4c1d95 0%, #2e1065 100%) !important; background-color: #4c1d95 !important;",
            "border_css": "border: 2px solid #d97706; box-shadow: 0 10px 25px rgba(109, 40, 217, 0.35);",
            "quote_color": "#fbbf24",
            "author_color": "#ffffff",
            "badge_bg": "#6d28d9",
            "badge_fg": "#fbbf24"
        },
        {
            "quote": "Simplicity is the ultimate sophistication.",
            "author": dev_author_input,
            "palette": DYNAMIC_POST_PALETTES[11],
            "font_style": "✍️ Montserrat (Modern Bold Sans)",
            "design_name": "🩶 Slate Gray & Gold",
            "bg_css": "background: linear-gradient(135deg, #1f2937 0%, #111827 100%) !important; background-color: #1f2937 !important;",
            "border_css": "border: 2px solid #d97706; box-shadow: 0 10px 25px rgba(17, 24, 39, 0.35);",
            "quote_color": "#f59e0b",
            "author_color": "#f9fafb",
            "badge_bg": "#374151",
            "badge_fg": "#f59e0b"
        }
    ]
    
    g_col1, g_col2, g_col3 = st.columns(3)
    for idx_g, sample_g in enumerate(gallery_samples):
        target_col = [g_col1, g_col2, g_col3][idx_g % 3]
        with target_col:
            p_buf_g = create_poster_card(
                sample_g['quote'],
                mood_text=sample_g['design_name'].upper(),
                author=sample_g['author'],
                palette=sample_g['palette'],
                font_style=sample_g['font_style']
            )
            # Display real crisp HD poster card image directly!
            st.image(p_buf_g, use_container_width=True)
            
            st.download_button(
                f"🖼️ Download Poster #{idx_g+1}",
                data=p_buf_g,
                file_name=f"inspire_ai_poster_design_{idx_g+1}.png",
                mime="image/png",
                use_container_width=True,
                key=f"gallery_dl_btn_{idx_g}"
            )

# ---------------------------------------------------------
# TAB 5: DL MODEL LAB
# ---------------------------------------------------------
with main_nav_tab5:
    st.markdown("### 🎓 Deep Learning Architecture & Tokenizer Lab")
    
    # Feature Badges
    st.markdown("""
        <div style="display: flex; gap: 8px; margin-bottom: 24px; flex-wrap: wrap; align-items: center;">
            <span class="badge-pill" style="font-weight: 800; border-color: #10b981; color: #10b981;">🟢 PLATFORM ONLINE</span>
            <span class="badge-pill">⚡ TensorFlow & Keras</span>
            <span class="badge-pill badge-pill-purple">🎨 Dedicated Poster Studio</span>
            <span class="badge-pill badge-pill-gold">🎮 AI Battle & E-Book</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 1. 🔤 Interactive Tokenizer Simulator")
    sim_input = st.text_input("Type any sentence to see how the model tokenizes it:", value="courage is the magic of life", key="tokenizer_sim_input")
    if sim_input.strip() and assets_loaded:
        seq_sim = tokenizer.texts_to_sequences([sim_input.lower()])[0]
        pad_sim = pad_sequences([seq_sim], maxlen=max_len, padding='pre')[0]
        
        c_tok1, c_tok2 = st.columns(2)
        with c_tok1:
            st.markdown("**Integer Token IDs:**")
            st.code(str(seq_sim), language="json")
        with c_tok2:
            st.markdown(f"**Padded Sequence (Max Length = {max_len}):**")
            st.code(str(list(pad_sim[-15:])), language="json")

    st.divider()
    st.markdown("#### 2. 🧠 LSTM Neural Network Architecture (3D Interactive Flip Grid)")
    
    flowchart_html = """
<style>
.flip-grid-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 16px 0;
}
@media (max-width: 900px) {
    .flip-grid-container {
        grid-template-columns: 1fr;
    }
}
.flip-card {
    background-color: transparent;
    width: 100%;
    height: 230px;
    perspective: 1000px;
}
.flip-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    text-align: center;
    transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    transform-style: preserve-3d;
    cursor: pointer;
    border-radius: 18px;
}
.flip-card:hover .flip-card-inner {
    transform: rotateY(180deg);
}
.flip-card-front, .flip-card-back {
    position: absolute;
    width: 100%;
    height: 100%;
    -webkit-backface-visibility: hidden;
    backface-visibility: hidden;
    border-radius: 18px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    box-sizing: border-box;
}
.flip-card-front {
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
}
.flip-card-back {
    transform: rotateY(180deg);
    background: #090d16 !important;
    border: 2px solid #fbbf24 !important;
    box-shadow: 0 10px 30px rgba(251, 191, 36, 0.25);
    text-align: left;
    align-items: flex-start;
}
</style>

<div class="flowchart-container" style="background: rgba(15, 23, 42, 0.95); border: 2px solid #38bdf8; border-radius: 24px; padding: 22px 18px; box-shadow: 0 15px 40px rgba(56, 189, 248, 0.2); margin: 16px 0;">
<div style="background: rgba(30, 41, 59, 0.95); border: 1.5px dashed #38bdf8; border-radius: 16px; padding: 12px 18px; margin-bottom: 16px;">
<div style="font-weight: 900; font-size: 0.92rem; color: #38bdf8 !important; -webkit-text-fill-color: #38bdf8 !important; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px;">🚀 End-to-End Live Data Transformation Flow</div>
<div class="flowchart-card-exp" style="font-size: 0.84rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; line-height: 1.5;">
<b style="color: #60a5fa !important; -webkit-text-fill-color: #60a5fa !important;">Input Seed:</b> <code style="background: #0f172a; color: #60a5fa !important; -webkit-text-fill-color: #60a5fa !important; padding: 2px 6px; border-radius: 4px; border: 1px solid #60a5fa;">"courage is"</code> ➔ 
<b style="color: #34d399 !important; -webkit-text-fill-color: #34d399 !important;">Tokens:</b> <code style="background: #0f172a; color: #34d399 !important; -webkit-text-fill-color: #34d399 !important; padding: 2px 6px; border-radius: 4px; border: 1px solid #34d399;">[0, ..., 264, 7]</code> ➔ 
<b style="color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important;">50D Vectors:</b> <code style="background: #0f172a; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; padding: 2px 6px; border-radius: 4px; border: 1px solid #fbbf24;">Semantic Space</code> ➔ 
<b style="color: #c084fc !important; -webkit-text-fill-color: #c084fc !important;">LSTM Memory:</b> <code style="background: #0f172a; color: #c084fc !important; -webkit-text-fill-color: #c084fc !important; padding: 2px 6px; border-radius: 4px; border: 1px solid #c084fc;">128 Units</code> ➔ 
<b style="color: #f87171 !important; -webkit-text-fill-color: #f87171 !important;">Softmax Word:</b> <code style="background: #0f172a; color: #f87171 !important; -webkit-text-fill-color: #f87171 !important; padding: 2px 6px; border-radius: 4px; border: 1px solid #f87171;">"the" (89.4%)</code>
</div>
</div>

<div class="flip-grid-container">
<!-- CARD 1 -->
<div class="flip-card">
<div class="flip-card-inner">
<div class="flip-card-front" style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1.5px solid #60a5fa;">
<span style="background: #2563eb; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;">Input Layer</span>
<div style="font-size: 2.2rem; margin: 4px 0;">🔤</div>
<div>
<div class="flowchart-card-title" style="font-weight: 900; font-size: 1.1rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;">1. Seed Prompt Input</div>
<div class="flowchart-card-sub" style="font-size: 0.8rem; color: #93c5fd !important; -webkit-text-fill-color: #93c5fd !important; margin-top: 4px;">User prompt (e.g., "courage is")</div>
</div>
<div style="font-size: 0.72rem; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; font-weight: 700;">🔄 Hover to flip for explanation</div>
</div>
<div class="flip-card-back">
<div style="font-size: 0.85rem; font-weight: 900; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; text-transform: uppercase; margin-bottom: 6px;">💡 Step 1 Explanation</div>
<div style="font-size: 0.82rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; line-height: 1.5;">
Receives the initial user prompt text to set the thematic context and starting seed direction for sequence generation.
</div>
<div style="margin-top: 10px; background: rgba(30, 58, 138, 0.5); padding: 6px 10px; border-radius: 8px; width: 100%; box-sizing: border-box;">
<span style="font-size: 0.75rem; color: #60a5fa !important; -webkit-text-fill-color: #60a5fa !important; font-weight: 800;">Data Output: Raw String Token</span>
</div>
</div>
</div>
</div>

<!-- CARD 2 -->
<div class="flip-card">
<div class="flip-card-inner">
<div class="flip-card-front" style="background: linear-gradient(135deg, rgba(6, 78, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1.5px solid #34d399;">
<span style="background: #059669; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;">Pre-processing</span>
<div style="font-size: 2.2rem; margin: 4px 0;">🔢</div>
<div>
<div class="flowchart-card-title" style="font-weight: 900; font-size: 1.1rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;">2. Tokenizer & Padding</div>
<div class="flowchart-card-sub" style="font-size: 0.8rem; color: #a7f3d0 !important; -webkit-text-fill-color: #a7f3d0 !important; margin-top: 4px;">Converts text ➔ Integer IDs</div>
</div>
<div style="font-size: 0.72rem; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; font-weight: 700;">🔄 Hover to flip for explanation</div>
</div>
<div class="flip-card-back">
<div style="font-size: 0.85rem; font-weight: 900; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; text-transform: uppercase; margin-bottom: 6px;">💡 Step 2 Explanation</div>
<div style="font-size: 0.82rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; line-height: 1.5;">
Converts raw words into integer IDs (e.g. 'courage' ➔ 264) and applies zero-padding to create fixed-length 20-word vectors.
</div>
<div style="margin-top: 10px; background: rgba(6, 78, 59, 0.5); padding: 6px 10px; border-radius: 8px; width: 100%; box-sizing: border-box;">
<span style="font-size: 0.75rem; color: #34d399 !important; -webkit-text-fill-color: #34d399 !important; font-weight: 800;">Data Output: 1D Integer Vector [20]</span>
</div>
</div>
</div>
</div>

<!-- CARD 3 -->
<div class="flip-card">
<div class="flip-card-inner">
<div class="flip-card-front" style="background: linear-gradient(135deg, rgba(120, 53, 15, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1.5px solid #fbbf24;">
<span style="background: #d97706; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;">50 Dimensions</span>
<div style="font-size: 2.2rem; margin: 4px 0;">🌐</div>
<div>
<div class="flowchart-card-title" style="font-weight: 900; font-size: 1.1rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;">3. Word Embedding</div>
<div class="flowchart-card-sub" style="font-size: 0.8rem; color: #fde68a !important; -webkit-text-fill-color: #fde68a !important; margin-top: 4px;">50D Semantic Vector Space</div>
</div>
<div style="font-size: 0.72rem; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; font-weight: 700;">🔄 Hover to flip for explanation</div>
</div>
<div class="flip-card-back">
<div style="font-size: 0.85rem; font-weight: 900; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; text-transform: uppercase; margin-bottom: 6px;">💡 Step 3 Explanation</div>
<div style="font-size: 0.82rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; line-height: 1.5;">
Maps discrete word IDs into a 50-dimensional continuous vector space where words with similar meanings cluster together.
</div>
<div style="margin-top: 10px; background: rgba(120, 53, 15, 0.5); padding: 6px 10px; border-radius: 8px; width: 100%; box-sizing: border-box;">
<span style="font-size: 0.75rem; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; font-weight: 800;">Data Output: 2D Matrix (20 x 50)</span>
</div>
</div>
</div>
</div>

<!-- CARD 4 -->
<div class="flip-card">
<div class="flip-card-inner">
<div class="flip-card-front" style="background: linear-gradient(135deg, rgba(131, 24, 67, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1.5px solid #f472b6;">
<span style="background: #db2777; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;">Rate: 20%</span>
<div style="font-size: 2.2rem; margin: 4px 0;">🛡️</div>
<div>
<div class="flowchart-card-title" style="font-weight: 900; font-size: 1.1rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;">4. Dropout Layer</div>
<div class="flowchart-card-sub" style="font-size: 0.8rem; color: #fbcfe8 !important; -webkit-text-fill-color: #fbcfe8 !important; margin-top: 4px;">Deactivates 20% Connections</div>
</div>
<div style="font-size: 0.72rem; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; font-weight: 700;">🔄 Hover to flip for explanation</div>
</div>
<div class="flip-card-back">
<div style="font-size: 0.85rem; font-weight: 900; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; text-transform: uppercase; margin-bottom: 6px;">💡 Step 4 Explanation</div>
<div style="font-size: 0.82rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; line-height: 1.5;">
Randomly mutes 20% of neuron paths during training to prevent memorization and ensure generalized text composition.
</div>
<div style="margin-top: 10px; background: rgba(131, 24, 67, 0.5); padding: 6px 10px; border-radius: 8px; width: 100%; box-sizing: border-box;">
<span style="font-size: 0.75rem; color: #f472b6 !important; -webkit-text-fill-color: #f472b6 !important; font-weight: 800;">Regularization: 0.2 Rate</span>
</div>
</div>
</div>
</div>

<!-- CARD 5 -->
<div class="flip-card">
<div class="flip-card-inner">
<div class="flip-card-front" style="background: linear-gradient(135deg, rgba(88, 28, 135, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1.5px solid #c084fc;">
<span style="background: #7c3aed; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;">128 Units</span>
<div style="font-size: 2.2rem; margin: 4px 0;">🧠</div>
<div>
<div class="flowchart-card-title" style="font-weight: 900; font-size: 1.1rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;">5. Deep LSTM Network</div>
<div class="flowchart-card-sub" style="font-size: 0.8rem; color: #e9d5ff !important; -webkit-text-fill-color: #e9d5ff !important; margin-top: 4px;">Recurrent Context Memory</div>
</div>
<div style="font-size: 0.72rem; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; font-weight: 700;">🔄 Hover to flip for explanation</div>
</div>
<div class="flip-card-back">
<div style="font-size: 0.85rem; font-weight: 900; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; text-transform: uppercase; margin-bottom: 6px;">💡 Step 5 Explanation</div>
<div style="font-size: 0.82rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; line-height: 1.5;">
Recurrent memory core containing 128 LSTM units with memory gates that learn long-term sequential dependencies.
</div>
<div style="margin-top: 10px; background: rgba(88, 28, 135, 0.5); padding: 6px 10px; border-radius: 8px; width: 100%; box-sizing: border-box;">
<span style="font-size: 0.75rem; color: #c084fc !important; -webkit-text-fill-color: #c084fc !important; font-weight: 800;">State Memory: 128 Hidden Units</span>
</div>
</div>
</div>
</div>

<!-- CARD 6 -->
<div class="flip-card">
<div class="flip-card-inner">
<div class="flip-card-front" style="background: linear-gradient(135deg, rgba(153, 27, 27, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1.5px solid #f87171;">
<span style="background: #dc2626; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;">8,978 Vocab</span>
<div style="font-size: 2.2rem; margin: 4px 0;">🎯</div>
<div>
<div class="flowchart-card-title" style="font-weight: 900; font-size: 1.1rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;">6. Softmax Output Layer</div>
<div class="flowchart-card-sub" style="font-size: 0.8rem; color: #fca5a5 !important; -webkit-text-fill-color: #fca5a5 !important; margin-top: 4px;">Probability Distribution</div>
</div>
<div style="font-size: 0.72rem; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; font-weight: 700;">🔄 Hover to flip for explanation</div>
</div>
<div class="flip-card-back">
<div style="font-size: 0.85rem; font-weight: 900; color: #fbbf24 !important; -webkit-text-fill-color: #fbbf24 !important; text-transform: uppercase; margin-bottom: 6px;">💡 Step 6 Explanation</div>
<div style="font-size: 0.82rem; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; line-height: 1.5;">
Dense Softmax layer computes probability scores across all 8,978 dictionary words to pick the best next word.
</div>
<div style="margin-top: 10px; background: rgba(153, 27, 27, 0.5); padding: 6px 10px; border-radius: 8px; width: 100%; box-sizing: border-box;">
<span style="font-size: 0.75rem; color: #f87171 !important; -webkit-text-fill-color: #f87171 !important; font-weight: 800;">Output: Softmax Probability Matrix</span>
</div>
</div>
</div>
</div>
</div>
</div>
"""
    st.markdown(flowchart_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 6: QUOTE BOOK CREATOR
# ---------------------------------------------------------
with main_nav_tab6:
    st.markdown("### 📖 InspireAI Official Quote Book Creator")
    st.markdown("Compile your saved favorite quotes into a stylized downloadable **HTML E-Book** file formatted for printing or viewing!")
    
    fav_list = st.session_state.get("favorites", [])
    if not fav_list:
        st.info("💡 You have 0 saved favorites. Adding sample quotes from dataset to your preview book...")
        if dataset_df is not None:
            q_col = 'quote' if 'quote' in dataset_df.columns else dataset_df.columns[0]
            fav_list = list(dataset_df[q_col].dropna().head(10))
        else:
            fav_list = ["The world is a canvas of infinite possibilities.", "Courage is not the absence of fear but the triumph over it."]

    st.markdown(f"Book Preview: **{len(fav_list)} Quotes Included**")
    for idx_b, q_b in enumerate(fav_list[:5]):
        st.markdown(f"- *“{q_b}”*")
    if len(fav_list) > 5:
        st.caption(f"...and {len(fav_list)-5} more quotes.")

    book_html_cards = "".join(f'<div class="card"><div class="quote">“{q}”</div><div class="author">— DEVELOPER: {dev_author_input.strip().upper()} —</div></div>' for q in fav_list)

    book_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>InspireAI — Official Quote Book</title>
    <style>
        body {{ font-family: 'Georgia', serif; background: #0f172a; color: #f8fafc; padding: 40px; line-height: 1.8; }}
        .cover {{ text-align: center; padding: 80px 20px; border-bottom: 2px solid #38bdf8; margin-bottom: 60px; }}
        .title {{ font-size: 3rem; color: #38bdf8; font-weight: bold; margin-bottom: 10px; }}
        .subtitle {{ font-size: 1.2rem; color: #94a3b8; letter-spacing: 2px; text-transform: uppercase; }}
        .card {{ background: #1e293b; border-left: 6px solid #818cf8; padding: 30px; margin-bottom: 30px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
        .quote {{ font-size: 1.6rem; color: #ffffff; font-weight: bold; margin-bottom: 15px; }}
        .author {{ font-size: 1rem; color: #fbbf24; text-transform: uppercase; font-weight: bold; letter-spacing: 1px; }}
    </style>
</head>
<body>
    <div class="cover">
        <div class="title">INSPIRE AI — QUOTE BOOK</div>
        <div class="subtitle">Compiled by Deep Learning LSTM Neural Studio</div>
        <p style="margin-top: 30px; color: #94a3b8;">Developer: {dev_author_input.strip().upper()}</p>
    </div>
    {book_html_cards}
</body>
</html>"""

    st.download_button(
        label="📖 Download Official Quote Book (.html)",
        data=book_html,
        file_name="inspire_ai_official_quote_book.html",
        mime="text/html",
        use_container_width=True,
        key="book_download_html_btn"
    )

# ---------------------------------------------------------
# TAB 7: ANALYTICS & DATASET
# ---------------------------------------------------------
with main_nav_tab7:
    st.markdown("### 📊 Dataset Analytics & Word Cloud")
    if dataset_df is not None:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("##### ☁️ Most Frequent Words in Quotes")
            try:
                all_text = " ".join(q for q in dataset_df['quote'].dropna())
                wc_bg = "#FFFFFF" if "Light Mode" in app_theme_choice else "#0F172A"
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color=wc_bg,
                    colormap="viridis" if "Light Mode" in app_theme_choice else "plasma",
                    max_words=100
                ).generate(all_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor(wc_bg)
                ax.set_facecolor(wc_bg)
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except Exception as err:
                st.error(f"WordCloud error: {err}")
                
        with c2:
            st.markdown("##### 🏆 Top Quoted Authors")
            if 'Author' in dataset_df.columns:
                top_authors = dataset_df['Author'].value_counts().head(8)
                st.bar_chart(top_authors)
            else:
                st.write("Author column not found.")
                
        st.divider()
        st.markdown(f"Total Quotes in Dataset: **{len(dataset_df)}**")
        search_query_ds = st.text_input("🔍 Search 3,000+ Quotes Dataset by Keyword or Author", "", key="dataset_explorer_search_input")
        if search_query_ds.strip():
            q_col = 'quote' if 'quote' in dataset_df.columns else dataset_df.columns[0]
            filtered_df = dataset_df[dataset_df[q_col].astype(str).str.contains(search_query_ds, case=False, na=False)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(dataset_df.head(50), use_container_width=True)
    else:
        st.info("Dataset CSV not found in workspace root.")

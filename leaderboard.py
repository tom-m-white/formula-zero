import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Formula Zero", layout="wide", page_icon="🏆")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv") 
        
        df = df.replace('N/A', np.nan)
        numeric_cols = [col for col in df.columns if col != 'Name']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        if "Standard elo" in df.columns:
            df = df.sort_values(by="Standard elo", ascending=False).reset_index(drop=True)
        
        df.insert(0, "Rank", df.index + 1)
        return df
        
    except FileNotFoundError:
        st.error("🚨 Could not find the CSV file! Please check the filename and make sure it is in the same folder as this script.")
        st.stop()

df = load_data()

st.markdown("""
    <h1 style='text-align: center; font-size: 3rem;'> Formula Zero Leaderboard </h1>
    <p style='text-align: center; color: gray;'> Updated 3/5/2026 </p>
    <hr>
""", unsafe_allow_html=True)

st.sidebar.header("🔍 Filters")
search_query = st.sidebar.text_input("Search Model Name:")
if search_query:
    df = df[df['Name'].str.contains(search_query, case=False)]

elo_cols = [col for col in df.columns if 'elo' in col.lower()]
win_cols = [col for col in df.columns if 'win %' in col.lower()]
dnf_cols = [col for col in df.columns if 'dnf %' in col.lower()]
lap_cols = [col for col in df.columns if 'lap' in col.lower()]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏅 Elo Ratings", "📈 Win Rates", "💥 DNF Rates", "⏱️ Avg Lap Times", "📊 Full Dataset"
])

def display_styled_df(data, subset_cols, cmap="Greens", format_str="{:.0f}"):
    display_df = data[['Rank', 'Name', 'Generations'] + subset_cols].copy()
    
    styled_df = display_df.style.background_gradient(
        cmap=cmap, subset=subset_cols
    ).format({col: format_str for col in subset_cols}, na_rep="N/A")
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)

with tab1:
    st.subheader("🏅 Elo Ratings by Difficulty")
    display_styled_df(df, elo_cols, cmap="Greens", format_str="{:.0f}")

with tab2:
    st.subheader("📈 Win Rates")
    display_styled_df(df, win_cols, cmap="Blues", format_str="{:.1%}")

with tab3:
    st.subheader("💥 Did Not Finish (DNF) Rates")
    display_styled_df(df, dnf_cols, cmap="Reds", format_str="{:.1%}")

with tab4:
    st.subheader("⏱️ Average Lap Times (Seconds)")
    display_styled_df(df, lap_cols, cmap="Oranges", format_str="{:.3f} s")

with tab5:
    st.subheader("📊 Complete Raw Dataset")
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("<hr><p style='text-align: center; color: gray;'> Made by Tom !</p>", unsafe_allow_html=True)
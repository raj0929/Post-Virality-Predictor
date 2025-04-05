from streamlit import (
    image, file_uploader, columns, sidebar,
    set_page_config, title, write, success,
    warning, info, error,spinner
)
from streamlit_option_menu import option_menu
import pandas as pd
import streamlit as st
import time
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



set_page_config(
    page_title="Post Virality Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

title_image,title_header = columns([1, 10])
with title_image:
    image("bar-graph.png", width=80)
with title_header:
    title("Post Virality Analyzer")
    write("_Predict and analyze your social media post performance_")

with sidebar:
    # title("Post Virality Analysis")
    # uploaded_file = file_uploader("Upload A CSV File", type=['csv'])
    title("🔍 Data Input")

    with st.expander("Upload Options", expanded=True):
        uploaded_file = file_uploader(
            "Choose a CSV file",
            type=['csv'],
            # help="Upload your social media metrics CSV file"
        )
    menu_tabs = None

    if uploaded_file is not None:
        with spinner("Loading data..."):
            try:
                df = pd.read_csv(uploaded_file)
                time.sleep(1)  # Simulate processing
                success("✅ Data loaded successfully!")

                menu_tabs = option_menu(
                    menu_title="📋 Menu",
                    options=['Dashboard', 'Analytics', 'Predictions'],
                    icons=['speedometer2', 'graph-up', 'magic'],
                    menu_icon="list",
                    default_index=0,
                )
            except Exception as e:
                error(f"❌ Error reading file: {str(e)}")
                df = pd.DataFrame()
    else:
        info("ℹ️ Please upload a CSV file to begin analysis")
        df = pd.DataFrame()
        menu_tabs = None


if menu_tabs:
    if menu_tabs == 'Dashboard':
        title("📊 Dashboard" )
        st.subheader("Data Preview")
        write(df.head(5))
        st.subheader("📈 Quick Insights")
        write(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns")

    if menu_tabs == "Analytics":
        content1, content2 = st.columns(2)
        with content1:
            st.subheader("📈 Posts Content")
            no_content_type = df["text_content_type"].unique()
            count_of_content_type = df["text_content_type"].value_counts()
            content_type = px.bar(df, x=no_content_type, y=count_of_content_type, color=no_content_type,
                                     title="Types of Content Posted")
            # content_type.update_xaxes(tickangle=45)
            st.plotly_chart(content_type, use_container_width=True)

        with content2:
            st.subheader("📉 Distribution of Platforms")
            platforms = df['platform'].unique()
            count_of_platform = df['platform'].value_counts()
            distribution_data = px.pie(df, names=platforms, values=count_of_platform, title='Platform Distribution',
                                       color=platforms, hole=0.3)
            st.plotly_chart(distribution_data, use_container_width=True)

        content3, content4 = st.columns(2)
        with content3:
            st.subheader("📉 Spread of Interactions")
            metrices = ['likes', 'shares', 'comments']
            df_melted = df.melt(value_vars=metrices,var_name="Metric", value_name="Count")
            interactions = px.box(df_melted, x='Metric',y='Count',title='Interactions', labels={"Count": "Engagement Count"},color="Metric")
            st.plotly_chart(interactions, use_container_width=True)

        with content4:
            st.subheader("📈 Enangement vs Virality Score")
            virality_status = px.scatter(df, x=df['previous_engagement'], y=df['virality_score'], color=df['viral'],
                                      title='Enangement vs Virality Score')
            st.plotly_chart(virality_status, use_container_width=True)

        st.subheader("🏆 Top 10 Posts with Highest Virality Score")
        data = pd.read_csv(uploaded_file, usecols=['meme_id','text_content_type','platform','likes','shares','comments','virality_score'])
        data_sorted = data.sort_values(by='virality_score',ascending=False).head(10)
        write(data_sorted)

        st.subheader("📊 Average Engagement by Platform")
        data1 = pd.read_csv(uploaded_file,
                           usecols=['platform', 'likes', 'shares', 'comments','virality_score'])
        platform_avg = data1.groupby('platform')[['likes','shares','comments','virality_score']].mean()
        write(platform_avg)

    if menu_tabs == 'Predictions':
        st.header("🔮 Virality Predictor")

        if not df.empty:
            le = LabelEncoder()
            df['text_content_type'] = le.fit_transform(df['text_content_type'])
            df['image_type'] = le.fit_transform(df['image_type'])
            df['platform'] = le.fit_transform(df['platform'])

            x = df[['text_content_type','image_type','platform','hashtags_used','previous_engagement']]
            y = df['viral']

            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(x_train,y_train)

            with st.form('Predict'):
                st.subheader("Make a Predciton")
                content = st.selectbox("Select a Content Type "
                                       "(0 - Dark Humor, 1 - Funny, 2 - Motivational, 3 - Political, 4 - Sarcastic)",options=df['text_content_type'].unique())
                image_mode = st.selectbox("Select a Image Type "
                                          "(0 - GIF, 1 - Image,2 - Video)",options=df['image_type'].unique())
                social_platform = st.selectbox("Select a Platform "
                                               "(0 - Facebook, 1 - Instagram, 2 - Reddit, 3 - TikTok, 4 - Twitter)",options=df['platform'].unique())
                hashtags = st.slider("How many Hashtags to use",0,20,5)
                engagement = st.number_input("What was engagement of previous post",value=1000)
                predict = st.form_submit_button("Predict Virality")

                if predict:
                    input_data = pd.DataFrame({
                        'text_content_type': [content],
                        'image_type': [image_mode],
                        'platform': [social_platform],
                        'hashtags_used': [hashtags],
                        'previous_engagement': [engagement]
                    })
                    prediction = model.predict(input_data)
                    if prediction[0]:
                        st.success("🎉This post is likely to go VIRAL")
                    else:
                        st.warning("⚠️ This post may not go viral")

else:
    warning("Please upload a CSV file to unlock all features")

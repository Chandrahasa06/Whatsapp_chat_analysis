import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data(show_spinner=False, persist="disk")
def load_data(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    return preprocessor.preprocess(data)

def load_css(css_path):
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def main():
    load_css("style.css")
    st.title('WhatsApp Chat Analyzer')
    uploaded_file = st.sidebar.file_uploader("Upload Exported Chat", type=["txt", "csv"])

    if uploaded_file:
        df = load_data(uploaded_file)

        user_list = df['user'].unique().tolist()
        user_list.sort()
        user_list.insert(0, "Overall")

        search_user = st.sidebar.text_input("Search for a User", "")
        filtered_users = [user for user in user_list if search_user.lower() in user.lower()]
        selected_user = st.sidebar.selectbox("Select The User", filtered_users)

        if selected_user == "Overall":
            analysis_menu = ["User Statistics", "Sentiment Analysis", "Advanced NLP Analysis", "Comparative Analysis", "User Activity", "Timeline Analysis"]
        else:
            analysis_menu = ["User Statistics", "Sentiment Analysis", "Advanced NLP Analysis", "User Activity", "Timeline Analysis"]

        st.sidebar.header("Analysis Options")
        choice = st.sidebar.selectbox("Select Analysis Type", analysis_menu, index=0)

        if choice == "Comparative Analysis":
            st.subheader("Comparative Analysis between Users")
            users_to_compare = st.multiselect("Select users for comparison", user_list)
            st.write("---")

            if users_to_compare:
                min_date = df["date"].min().date()
                max_date = df["date"].max().date()
                selected_range = st.slider("Select Time Range", min_date, max_date, (min_date, max_date))
                st.write("---")

                if st.sidebar.button("Show Comparative Analysis", key="comparative_analysis_button"):
                    users_activity = helper.perform_comparative_analysis(df, users_to_compare, selected_range[0], selected_range[1])
                    st.bar_chart(users_activity)

        elif st.sidebar.button("Start Analysis"):
            if choice == "User Statistics":
                stats = helper.fetch_stats(selected_user, df)
                labels = [
                    "Total Messages", "Total Words", "Total Media", "Total Links",
                    "Total Emojis", "Deleted Messages", "Edited Messages",
                    "Shared Contacts", "Shared Locations"
                ]
                for label, value in zip(labels, stats):
                    st.markdown(f"### {label}:")
                    st.write(f"<div class='big-font'>{value}</div>", unsafe_allow_html=True)
                    st.write("---")

            elif choice == "Sentiment Analysis":
                if selected_user != 'Overall':
                    df = df[df['user'] == selected_user]
                df['Sentiment'] = df['message'].apply(helper.extract_sentiment)

                st.subheader("Sentiment Distribution")
                fig = px.bar(df['Sentiment'].value_counts(), labels={'index': 'Sentiment', 'value': 'Count'})
                st.plotly_chart(fig)
                st.write("---")

                if 'date' in df.columns:
                    st.subheader("Sentiment Trends Over Time")
                    sentiment_over_time = df.groupby(['date', 'Sentiment']).size().reset_index(name='Counts')
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.lineplot(data=sentiment_over_time, x='date', y='Counts', hue='Sentiment', ax=ax)
                    st.pyplot(fig)

            elif choice == "Advanced NLP Analysis":
                st.subheader("TF-IDF Analysis")
                top_words = helper.perform_tfidf_analysis(df['message'])
                st.write("Top 5 words:", top_words)

                st.subheader("LDA Topic Modeling")
                topics = helper.perform_lda_analysis(df['message'], 5)
                for topic in topics:
                    st.write(topic)

            elif choice == "User Activity":
                if selected_user == 'Overall':
                    top, bottom = helper.most_least_busy_users(df)
                    st.subheader("Most Active Users")
                    st.bar_chart(top)
                    st.subheader("Least Active Users")
                    st.bar_chart(bottom)
                else:
                    st.subheader("User Activity Over Time")
                    act = helper.user_activity_over_time(selected_user, df)
                    st.line_chart(act)

                # st.subheader("Week Activity Map")
                # week_data = helper.week_activity_map(selected_user, df)
                # fig, ax = plt.subplots(figsize=(8, 6))
                # week_data.sort_index().plot(kind='bar', ax=ax)
                # st.pyplot(fig)

                st.subheader("Month Activity Map")
                month_data = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots(figsize=(8, 6))
                month_data.sort_index().plot(kind='bar', ax=ax)
                st.pyplot(fig)

                st.subheader("Activity Heatmap")
                heat = helper.activity_heatmap(selected_user, df)
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(heat, cmap='viridis', annot=True, fmt=".0f", ax=ax)
                st.pyplot(fig)

            # elif choice == "Word and Emoji Analysis":
            #     wc_image = helper.create_wordcloud(selected_user, df)
            #     with st.container():
            #         st.subheader("Word Cloud")
            #         st.image(wc_image, use_container_width=True)

            #         st.subheader("Emoji Analysis")
            #         emoji_df = helper.emoji_helper(selected_user, df)
            #         st.dataframe(emoji_df.head())

            #         if not emoji_df.empty:
            #             fig, ax = plt.subplots(figsize=(5, 5))  # Keep it compact
            #             ax.pie(
            #                 emoji_df['Frequency'].head(),
            #                 labels=emoji_df['Emoji'].head(),
            #                 autopct='%1.1f%%',
            #                 startangle=90
            #             )
            #             ax.axis('equal')  # Equal aspect ratio ensures pie is a circle.
            #             plt.tight_layout()  # Prevent cutting off
            #             st.pyplot(fig)


            elif choice == "Timeline Analysis":
                st.subheader("Monthly Timeline")
                monthly_df = helper.monthly_timeline(selected_user, df)
                st.line_chart(monthly_df.set_index('time')['message'])

                st.subheader("Daily Timeline")
                daily_df = helper.daily_timeline(selected_user, df)
                st.line_chart(daily_df.set_index('only_date')['message'])

    # Feedback Section
    st.sidebar.markdown("---")
    st.sidebar.header("We Value Your Feedback")
    was_helpful = st.sidebar.selectbox("Was this useful?", ["Please choose", "Yes", "Somewhat", "No"])
    if was_helpful != "Please choose":
        comment = st.sidebar.text_area("Any suggestions?")
        if st.sidebar.button("Submit Feedback"):
            st.sidebar.success("Thank you for your feedback!")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()

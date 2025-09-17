import streamlit as st
import pandas as pd
import altair as alt
import glob
import io

st.set_page_config(page_title="Top Tags Dashboard", layout="wide")

st.title("Top Tags — Deep Dive")

@st.cache_data
def load_top_tags(path="top_tags_summary.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File not found: {path}")
        return pd.DataFrame(columns=["Conversation tag", "New conversations"])    
    return df

df = load_top_tags()

if df.empty:
    st.info("No data found in `top_tags_summary.csv`. Run the data pipeline first.")
else:
    tabs = st.tabs(["Top Tags", "Deep Dive"])

    # --- Top Tags tab ---
    with tabs[0]:
        # Allow user to pick N
        max_n = min(50, len(df))
        n = st.sidebar.slider("Top N tags", min_value=1, max_value=max_n, value=min(5, max_n))

        # Determine columns
        cols = df.columns.tolist()
        # Try to find count and tag columns
        count_col = next((c for c in cols if 'new' in c.lower() or 'count' in c.lower()), cols[-1])
        tag_col = next((c for c in cols if 'tag' in c.lower() or 'conversation' in c.lower()), cols[0])

        df_sorted = df.sort_values(by=count_col, ascending=False).head(n)

        st.subheader(f"Top {n} most used tags")
        chart = alt.Chart(df_sorted).mark_bar().encode(
            x=alt.X(f"{count_col}:Q", title="Count"),
            y=alt.Y(f"{tag_col}:N", sort='-x', title="Tag"),
            tooltip=[tag_col, count_col]
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

        st.write(df_sorted[[tag_col, count_col]].reset_index(drop=True))

    # --- Deep Dive tab ---
    with tabs[1]:
        st.subheader("Per-tag deep dive")
        # Find available per-tag files in the project root
        kw_files = glob.glob('*_question_keywords.csv') + glob.glob('*/*_question_keywords.csv')
        q_files = glob.glob('*_user_questions.csv') + glob.glob('*/*_user_questions.csv')

        # Derive available tags from filenames (prefix before _question_keywords)
        tags = []
        for f in sorted(set(kw_files)):
            base = f.split('/')[-1]
            if base.endswith('_question_keywords.csv'):
                prefix = base.replace('_question_keywords.csv', '')
                tags.append(prefix)

        if not tags:
            st.info('No per-tag CSVs found (e.g. My_Account_question_keywords.csv). Run the deep dive pipeline first.')
        else:
                # Create a dynamic tab per tag so users can switch between categories quickly
                tag_tabs = st.tabs([t.replace('_', ' ') for t in tags])
                for i, prefix in enumerate(tags):
                    with tag_tabs[i]:
                        st.header(prefix.replace('_', ' '))
                        # Load keyword frequencies
                        kw_path = next((p for p in kw_files if p.endswith(f"{prefix}_question_keywords.csv")), None)
                        if kw_path:
                            kw_df = pd.read_csv(kw_path)
                            if 'frequency' in kw_df.columns:
                                kw_count_col = 'frequency'
                                kw_keyword_col = 'keyword'
                            elif kw_df.shape[1] >= 2:
                                kw_keyword_col, kw_count_col = kw_df.columns[:2]
                            else:
                                kw_keyword_col = kw_df.columns[0]
                                kw_count_col = None

                            # Immediately generate and display the word cloud
                            try:
                                from wordcloud import WordCloud
                            except Exception:
                                WordCloud = None

                            if WordCloud is None:
                                st.warning('The `wordcloud` package is not available in the current environment.')
                            else:
                                if kw_count_col:
                                    freqs = {str(k): int(v) for k, v in zip(kw_df[kw_keyword_col].astype(str), kw_df[kw_count_col].astype(int))}
                                    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freqs)
                                else:
                                    text = ' '.join(kw_df[kw_keyword_col].astype(str).tolist())
                                    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

                                buf = io.BytesIO()
                                wc.to_image().save(buf, format='PNG')
                                buf.seek(0)
                                st.image(buf, use_container_width=True)

                        # Load questions
                        q_path = next((p for p in q_files if p.endswith(f"{prefix}_user_questions.csv")), None)
                        if q_path:
                            q_df = pd.read_csv(q_path)
                            st.write(f"Showing {len(q_df)} user questions:")
                            st.write(q_df.head(200))
                            q_csv = q_df.to_csv(index=False)
                            st.download_button('Download questions CSV', data=q_csv, file_name=f'{prefix}_user_questions.csv')

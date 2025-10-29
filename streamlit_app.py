import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from name_matcher import get_name_matcher
import time

# Page configuration
st.set_page_config(
    page_title="Name Matcher Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .match-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .score-high { background: linear-gradient(135deg, #4CAF50, #45a049); }
    .score-medium { background: linear-gradient(135deg, #FF9800, #F57C00); }
    .score-low { background: linear-gradient(135deg, #FF5722, #D84315); }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #4ECDC4;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'matcher' not in st.session_state:
        st.session_state.matcher = get_name_matcher()

def get_score_color(score):
    """Return color based on similarity score"""
    if score >= 0.8:
        return "score-high"
    elif score >= 0.5:
        return "score-medium"
    else:
        return "score-low"

def display_match_card(name, score, rank):
    """Display a match card with ranking"""
    color_class = get_score_color(score)
    st.markdown(f"""
    <div class="match-card {color_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin:0; font-size: 1.4rem;">#{rank} {name}</h3>
                <p style="margin:0; opacity: 0.9;">Similarity Score: {score:.4f}</p>
            </div>
            <div style="font-size: 2rem; opacity: 0.8;">
                {"🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def plot_similarity_scores(matches):
    """Create a bar chart of similarity scores"""
    df = pd.DataFrame(matches, columns=['Name', 'Score'])
    df['Rank'] = range(1, len(df) + 1)
    
    fig = px.bar(
        df, 
        x='Score', 
        y='Name', 
        orientation='h',
        title='Name Similarity Scores',
        color='Score',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        showlegend=False
    )
    
    return fig

def plot_score_distribution(matches):
    """Create a distribution plot of scores"""
    scores = [score for _, score in matches]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color='#4ECDC4',
        opacity=0.7,
        name='Score Distribution'
    ))
    
    fig.update_layout(
        title='Similarity Score Distribution',
        xaxis_title='Similarity Score',
        yaxis_title='Frequency',
        height=300
    )
    
    return fig

def display_metrics(best_match, all_matches, method):
    """Display key metrics"""
    best_name, best_score = best_match
    avg_score = sum(score for _, score in all_matches) / len(all_matches)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Best Match Score",
            value=f"{best_score:.4f}",
            delta="Perfect" if best_score == 1.0 else "High" if best_score > 0.8 else "Medium"
        )
    
    with col2:
        st.metric(
            label="Average Score",
            value=f"{avg_score:.4f}",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            label="Matches Found",
            value=str(len(all_matches))
        )
    
    with col4:
        st.metric(
            label="Method Used",
            value=method.title()
        )

def sidebar_controls():
    """Create sidebar controls"""
    st.sidebar.title("⚙️ Configuration")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Similarity Method",
        ["combined", "sequence", "levenshtein", "tfidf"],
        index=0,
        help="Choose the algorithm for name matching"
    )
    
    # Number of results
    top_k = st.sidebar.slider(
        "Number of Results",
        min_value=5,
        max_value=20,
        value=10,
        help="How many matches to display"
    )
    
    # Dataset info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"Total names: {len(st.session_state.matcher.names)}")
    
    # Add new name
    st.sidebar.markdown("---")
    st.sidebar.subheader("➕ Add New Name")
    new_name = st.sidebar.text_input("Enter new name to add to dataset")
    if st.sidebar.button("Add Name") and new_name:
        if new_name not in st.session_state.matcher.names:
            st.session_state.matcher.add_name(new_name)
            st.sidebar.success(f"✅ '{new_name}' added to dataset!")
        else:
            st.sidebar.warning(f"⚠️ '{new_name}' already exists in dataset")
    
    return method, top_k

def main():
    """Main Streamlit application"""
    load_css()
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Name Matcher Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Find similar names using advanced matching algorithms")
    
    # Sidebar
    method, top_k = sidebar_controls()
    
    # Main search area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input(
            "🔤 Enter a name to search",
            placeholder="e.g., Geetha, John, Michael...",
            help="Type a name to find similar matches from our dataset"
        )
    
    with col2:
        st.write("")  # Spacing
        search_clicked = st.button("🚀 Find Matches", use_container_width=True)
    
    # Perform search
    if search_clicked and query:
        with st.spinner('🔍 Searching for matches...'):
            # Simulate loading for better UX
            time.sleep(0.5)
            
            results = st.session_state.matcher.find_matches(
                query, 
                method=method, 
                top_k=top_k
            )
            
            # Store in history
            st.session_state.search_history.append({
                'query': query,
                'method': method,
                'timestamp': time.time(),
                'results': results
            })
            
            # Display results
            if results['best_match'][0]:
                best_match = results['best_match']
                all_matches = results['all_matches']
                
                # Display metrics
                display_metrics(best_match, all_matches, method)
                
                st.markdown("---")
                
                # Results in two columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🏆 Top Matches")
                    for i, (name, score) in enumerate(all_matches, 1):
                        display_match_card(name, score, i)
                
                with col2:
                    st.subheader("📈 Visualizations")
                    
                    # Similarity scores chart
                    fig1 = plot_similarity_scores(all_matches)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Score distribution
                    fig2 = plot_score_distribution(all_matches)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Detailed table
                st.markdown("---")
                st.subheader("📋 Detailed Results")
                df = pd.DataFrame(all_matches, columns=['Name', 'Similarity Score'])
                df['Rank'] = range(1, len(df) + 1)
                df['Similarity Score'] = df['Similarity Score'].round(4)
                st.dataframe(df.set_index('Rank'), use_container_width=True)
                
            else:
                st.error("❌ No matches found. Try a different name or method.")
    
    # Search history
    if st.session_state.search_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Search History")
        
        for i, history in enumerate(reversed(st.session_state.search_history[-5:])):
            query = history['query']
            method = history['method']
            best_match = history['results']['best_match'][0]
            best_score = history['results']['best_match'][1]
            
            st.sidebar.write(f"**{query}** → {best_match} ({best_score:.3f})")
    
    # Instructions
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        ### Usage Guide
        
        1. **Enter a name** in the search box above
        2. **Choose a method** from the sidebar:
           - **Combined**: Best overall accuracy (recommended)
           - **Sequence**: Based on character sequence matching
           - **Levenshtein**: Based on edit distance
           - **TF-IDF**: Based on character n-grams
        
        3. **Adjust results** using the slider for more/less matches
        4. **Analyze results** through visualizations and detailed tables
        
        ### Example Queries
        - `Geetha` → Finds: Geeta, Gita, Githa
        - `Jon` → Finds: John, Jonathan, Johny
        - `Katherine` → Finds: Kate, Katie, Catherine
        """)
    
    # Method comparison
    with st.expander("🔬 Compare Methods"):
        st.subheader("Method Comparison")
        compare_name = st.text_input("Enter name to compare methods", "Geetha")
        
        if st.button("Compare Methods") and compare_name:
            methods = ["combined", "sequence", "levenshtein", "tfidf"]
            comparison_data = []
            
            for method in methods:
                results = st.session_state.matcher.find_matches(compare_name, method=method, top_k=1)
                best_match, best_score = results['best_match']
                comparison_data.append({
                    'Method': method.title(),
                    'Best Match': best_match,
                    'Score': best_score
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()
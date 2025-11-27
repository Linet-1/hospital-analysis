import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Hospital Data Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced safe functions
def create_safe_bar_chart(data, x_col, y_col, title, orientation='v', color_col=None, **kwargs):
    """
    Universal safe bar chart function that handles ALL possible data scenarios
    """
    try:
        # Scenario 1: Empty data
        if data is None or data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No dataset available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, 
                              showarrow=False, font=dict(size=14))
            fig.update_layout(title=title, showlegend=False)
            return fig
        
        # Scenario 2: Series input (the main error case)
        if hasattr(data, 'name') and hasattr(data, 'index'):
            # Convert Series to DataFrame
            plot_df = data.reset_index()
            if len(plot_df.columns) == 2:
                x_col = plot_df.columns[1] if orientation == 'h' else plot_df.columns[0]
                y_col = plot_df.columns[0] if orientation == 'h' else plot_df.columns[1]
            else:
                plot_df = pd.DataFrame({y_col: data.values, x_col: data.index})
        
        # Scenario 3: Proper DataFrame
        else:
            plot_df = data.copy()
        
        # Ensure columns exist
        if x_col not in plot_df.columns or y_col not in plot_df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No dataset available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, 
                              showarrow=False, font=dict(size=12))
            fig.update_layout(title=title, showlegend=False)
            return fig
        
        # Remove any NaN values that might cause issues
        plot_df = plot_df.dropna(subset=[x_col, y_col])
        
        if plot_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No dataset available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, 
                              showarrow=False, font=dict(size=12))
            fig.update_layout(title=title, showlegend=False)
            return fig
        
        # Create the actual chart
        if orientation == 'h':
            fig = px.bar(
                plot_df,
                y=x_col,
                x=y_col,
                title=title,
                color=color_col,
                orientation='h',
                **kwargs
            )
        else:
            fig = px.bar(
                plot_df,
                x=x_col,
                y=y_col,
                title=title,
                color=color_col,
                **kwargs
            )
        
        return fig
        
    except Exception as e:
        # Ultimate fallback for ANY error
        fig = go.Figure()
        fig.add_annotation(text="No dataset available", 
                          xref="paper", yref="paper", x=0.5, y=0.5, 
                          showarrow=False, font=dict(size=12))
        fig.update_layout(title=title, showlegend=False)
        return fig

def safe_aggregate_data(df, group_col, value_col, agg_func='mean', min_count=1):
    """
    Safely aggregate data with comprehensive error handling
    """
    try:
        if df is None or df.empty or group_col not in df.columns or value_col not in df.columns:
            return pd.DataFrame()
        
        aggregated = df.groupby(group_col)[value_col].agg([agg_func, 'count']).reset_index()
        aggregated = aggregated[aggregated['count'] >= min_count]
        aggregated.columns = [group_col, value_col, 'count']
        
        return aggregated.sort_values(value_col, ascending=False).head(10)
        
    except Exception:
        return pd.DataFrame()

def create_safe_pie_chart(data, names_col, values_col, title=""):
    """Create pie chart with empty data handling"""
    try:
        if data is None or data.empty or names_col not in data.columns or values_col not in data.columns:
            fig = go.Figure()
            fig.add_annotation(text="No dataset available",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, xanchor='center', yanchor='middle',
                              showarrow=False, font=dict(size=14))
            fig.update_layout(title=title)
            return fig
        
        # Ensure we have data to plot
        if len(data) == 0:
            raise ValueError("No data after filtering")
            
        fig = px.pie(data, names=names_col, values=values_col, title=title, hole=0.4)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text="No dataset available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(title=title)
        return fig

def create_safe_histogram(data, x_col, title="", **kwargs):
    """Create histogram with empty data handling"""
    try:
        if data is None or data.empty or x_col not in data.columns:
            fig = go.Figure()
            fig.add_annotation(text="No dataset available",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, xanchor='center', yanchor='middle',
                              showarrow=False, font=dict(size=14))
            fig.update_layout(title=title)
            return fig
        
        fig = px.histogram(data, x=x_col, title=title, **kwargs)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text="No dataset available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(title=title)
        return fig

def create_safe_scatter(data, x_col, y_col, color_col=None, title="", **kwargs):
    """Create scatter plot with empty data handling"""
    try:
        if data is None or data.empty or x_col not in data.columns or y_col not in data.columns:
            fig = go.Figure()
            fig.add_annotation(text="No dataset available",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, xanchor='center', yanchor='middle',
                              showarrow=False, font=dict(size=14))
            fig.update_layout(title=title)
            return fig
        
        if color_col and color_col in data.columns:
            fig = px.scatter(data, x=x_col, y=y_col, color=color_col, title=title, **kwargs)
        else:
            fig = px.scatter(data, x=x_col, y=y_col, title=title, **kwargs)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text="No dataset available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(title=title)
        return fig

def safe_display_metric(label, value, format_str="{}", default_display="N/A"):
    """Safely display metrics with NaN handling"""
    try:
        if pd.isna(value) or value is None:
            st.metric(label, default_display)
        else:
            formatted_value = format_str.format(value)
            st.metric(label, formatted_value)
    except:
        st.metric(label, default_display)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Try multiple file locations
        file_paths = [
            "hospital data analysis.csv",
            r"C:\Users\Zyon\Downloads\hospital data analysis.csv",
            "./hospital data analysis.csv"
        ]
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                st.success(f"✅ Data loaded successfully from: {file_path}")
                return df
            except FileNotFoundError:
                continue
        
        # If file not found, use file uploader
        st.warning("📁 CSV file not found in default locations. Please upload your hospital data file.")
        uploaded_file = st.file_uploader("Choose hospital data CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            return df
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def main():
    st.title("🏥 Hospital Data Analysis Dashboard")
    st.markdown("Analyzing patient treatment patterns, costs, and outcomes")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.info("💡 Please make sure your CSV file is available or upload it using the file uploader.")
        return
    
    # Data preprocessing
    try:
        # Clean column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Convert to proper data types
        numeric_columns = ['age', 'cost', 'satisfaction', 'length_of_stay']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create visit date for time analysis
        df['visit_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        
    except Exception as e:
        st.error(f"❌ Error in data preprocessing: {str(e)}")
        return
    
    # Sidebar - Dataset Information
    st.sidebar.header("📊 Dataset Overview")
    safe_display_metric("Total Patients", len(df), "{:,}", "N/A")
    safe_display_metric("Total Records", len(df), "{:,}", "N/A")
    
    unique_conditions = df['condition'].nunique() if 'condition' in df.columns else 0
    safe_display_metric("Unique Conditions", unique_conditions, "{}", "N/A")
    
    if 'visit_date' in df.columns:
        date_range = f"{df['visit_date'].min().strftime('%Y-%m-%d')} to {df['visit_date'].max().strftime('%Y-%m-%d')}"
        st.sidebar.metric("Date Range", date_range)
    else:
        st.sidebar.metric("Date Range", "N/A")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Medical Condition filter
    condition_options = ['All'] + sorted(df['condition'].unique()) if 'condition' in df.columns else ['All']
    selected_condition = st.sidebar.selectbox("Medical Condition", condition_options)
    
    # Gender filter
    gender_options = ['All'] + list(df['gender'].unique()) if 'gender' in df.columns else ['All']
    selected_gender = st.sidebar.selectbox("Gender", gender_options)
    
    # Outcome filter
    outcome_options = ['All'] + list(df['outcome'].unique()) if 'outcome' in df.columns else ['All']
    selected_outcome = st.sidebar.selectbox("Outcome", outcome_options)
    
    # Readmission filter
    readmission_options = ['All'] + list(df['readmission'].unique()) if 'readmission' in df.columns else ['All']
    selected_readmission = st.sidebar.selectbox("Readmission", readmission_options)
    
    # Age range filter
    if 'age' in df.columns:
        min_age, max_age = int(df['age'].min()), int(df['age'].max())
    else:
        min_age, max_age = 0, 100
    age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
    
    # Cost range filter
    if 'cost' in df.columns:
        min_cost, max_cost = int(df['cost'].min()), int(df['cost'].max())
    else:
        min_cost, max_cost = 0, 10000
    cost_range = st.sidebar.slider("Cost Range ($)", min_cost, max_cost, (min_cost, max_cost))
    
    # Apply filters
    filtered_df = df.copy()
    if selected_condition != 'All':
        filtered_df = filtered_df[filtered_df['condition'] == selected_condition]
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['gender'] == selected_gender]
    if selected_outcome != 'All':
        filtered_df = filtered_df[filtered_df['outcome'] == selected_outcome]
    if selected_readmission != 'All':
        filtered_df = filtered_df[filtered_df['readmission'] == selected_readmission]
    
    # Apply numeric filters safely
    try:
        filtered_df = filtered_df[
            (filtered_df['age'] >= age_range[0]) & 
            (filtered_df['age'] <= age_range[1]) &
            (filtered_df['cost'] >= cost_range[0]) & 
            (filtered_df['cost'] <= cost_range[1])
        ]
    except:
        # If any column doesn't exist, just use the existing filtered_df
        pass
    
    # KPI Metrics - Row 1
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(filtered_df) if filtered_df is not None else 0
        safe_display_metric("Total Patients", total_patients, "{:,}", "N/A")
    
    with col2:
        avg_cost = filtered_df['cost'].mean() if filtered_df is not None and 'cost' in filtered_df.columns else float('nan')
        safe_display_metric("Average Cost", avg_cost, "${:,.2f}", "N/A")
    
    with col3:
        avg_stay = filtered_df['length_of_stay'].mean() if filtered_df is not None and 'length_of_stay' in filtered_df.columns else float('nan')
        safe_display_metric("Avg Length of Stay", avg_stay, "{:.1f} days", "N/A")
    
    with col4:
        if filtered_df is not None and 'readmission' in filtered_df.columns:
            readmission_rate = (filtered_df['readmission'] == 'Yes').mean() * 100
        else:
            readmission_rate = float('nan')
        safe_display_metric("Readmission Rate", readmission_rate, "{:.1f}%", "N/A")
    
    # KPI Metrics - Row 2
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        total_cost = filtered_df['cost'].sum() if filtered_df is not None and 'cost' in filtered_df.columns else float('nan')
        safe_display_metric("Total Cost", total_cost, "${:,.0f}", "N/A")
    
    with col6:
        avg_satisfaction = filtered_df['satisfaction'].mean() if filtered_df is not None and 'satisfaction' in filtered_df.columns else float('nan')
        safe_display_metric("Avg Satisfaction", avg_satisfaction, "{:.1f}/5", "N/A")
    
    with col7:
        if filtered_df is not None and 'outcome' in filtered_df.columns:
            recovered_rate = (filtered_df['outcome'] == 'Recovered').mean() * 100
        else:
            recovered_rate = float('nan')
        safe_display_metric("Recovery Rate", recovered_rate, "{:.1f}%", "N/A")
    
    with col8:
        procedure_count = filtered_df['procedure'].nunique() if filtered_df is not None and 'procedure' in filtered_df.columns else 0
        safe_display_metric("Procedures Used", procedure_count, "{}", "N/A")
    
    st.markdown("---")
    
    # Visualizations Section
    st.subheader("📊 Treatment Pattern Analysis")
    
    # Create tabs for organized visualization
    tab1, tab2, tab3, tab4 = st.tabs(["Cost Analysis", "Patient Demographics", "Treatment Patterns", "Outcome Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost Distribution
            st.subheader("💰 Treatment Cost Distribution")
            fig_cost_hist = create_safe_histogram(
                filtered_df,
                x_col='cost',
                title="Distribution of Treatment Costs",
                nbins=20,
                color_discrete_sequence=['#1f77b4']
            )
            fig_cost_hist.update_layout(xaxis_title="Cost ($)", yaxis_title="Number of Patients")
            st.plotly_chart(fig_cost_hist, use_container_width=True)
            
        with col2:
            # Most Expensive Procedures
            st.subheader("💊 Most Expensive Procedures")
            procedure_data = safe_aggregate_data(filtered_df, 'procedure', 'cost', 'mean')
            
            fig_procedures = create_safe_bar_chart(
                procedure_data,
                x_col='procedure',
                y_col='cost',
                title="Top 10 Most Expensive Procedures",
                color_col='cost',
                color_continuous_scale='Viridis'
            )
            fig_procedures.update_layout(xaxis_title="Procedure", yaxis_title="Average Cost ($)")
            st.plotly_chart(fig_procedures, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age Distribution
            st.subheader("👥 Patient Age Distribution")
            fig_age = create_safe_histogram(
                filtered_df,
                x_col='age',
                title="Distribution of Patient Ages",
                nbins=20,
                color_discrete_sequence=['#2ecc71']
            )
            fig_age.update_layout(xaxis_title="Age", yaxis_title="Number of Patients")
            st.plotly_chart(fig_age, use_container_width=True)
            
        with col2:
            # Gender Distribution
            st.subheader("🚻 Gender Distribution")
            if filtered_df is not None and 'gender' in filtered_df.columns:
                gender_counts = filtered_df['gender'].value_counts().reset_index()
                gender_counts.columns = ['gender', 'count']
                
                fig_gender = create_safe_pie_chart(
                    gender_counts,
                    names_col='gender',
                    values_col='count',
                    title="Gender Distribution of Patients"
                )
                st.plotly_chart(fig_gender, use_container_width=True)
            else:
                st.info("No dataset available")
            
            # Gender by Condition
            st.subheader("👥 Gender Distribution by Condition")
            if filtered_df is not None and all(col in filtered_df.columns for col in ['condition', 'gender']):
                gender_condition = pd.crosstab(filtered_df['condition'], filtered_df['gender']).head(10).reset_index()
                gender_condition_melted = gender_condition.melt(id_vars=['condition'], var_name='gender', value_name='count')
                
                fig_gender_cond = create_safe_bar_chart(
                    gender_condition_melted,
                    x_col='condition',
                    y_col='count',
                    title="Top 10 Conditions by Gender",
                    color_col='gender'
                )
                st.plotly_chart(fig_gender_cond, use_container_width=True)
            else:
                st.info("No dataset available")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Medical Conditions
            st.subheader("🩺 Top Medical Conditions")
            if filtered_df is not None and 'condition' in filtered_df.columns:
                condition_data = filtered_df['condition'].value_counts().head(10).reset_index()
                condition_data.columns = ['condition', 'count']
                
                fig_conditions = create_safe_bar_chart(
                    condition_data,
                    x_col='count',
                    y_col='condition',
                    title="Top 10 Most Common Conditions",
                    color_col='count',
                    orientation='h',
                    color_continuous_scale='Blues'
                )
                fig_conditions.update_layout(
                    xaxis_title="Number of Patients",
                    yaxis_title="Medical Condition"
                )
                st.plotly_chart(fig_conditions, use_container_width=True)
            else:
                st.info("No dataset available")
            
        with col2:
            # Cost by Condition
            st.subheader("💰 Average Cost by Condition")
            cost_condition_data = safe_aggregate_data(filtered_df, 'condition', 'cost', 'mean')
            
            fig_cost_condition = create_safe_bar_chart(
                cost_condition_data,
                x_col='cost',
                y_col='condition',
                title="Top 10 Conditions by Average Cost",
                color_col='cost',
                orientation='h',
                color_continuous_scale='Reds'
            )
            fig_cost_condition.update_layout(xaxis_title="Average Cost ($)", yaxis_title="Medical Condition")
            st.plotly_chart(fig_cost_condition, use_container_width=True)
        
        # Length of Stay vs Cost
        st.subheader("📊 Treatment Duration vs Cost Analysis")
        fig_scatter = create_safe_scatter(
            filtered_df,
            x_col='length_of_stay',
            y_col='cost',
            color_col='condition',
            title="Relationship between Length of Stay and Treatment Cost",
            hover_data=['gender', 'procedure', 'outcome'] if filtered_df is not None else None
        )
        fig_scatter.update_layout(xaxis_title="Length of Stay (Days)", yaxis_title="Treatment Cost ($)")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Patient Outcomes
            st.subheader("📈 Patient Outcomes")
            if filtered_df is not None and 'outcome' in filtered_df.columns:
                outcome_counts = filtered_df['outcome'].value_counts().reset_index()
                outcome_counts.columns = ['outcome', 'count']
                
                fig_outcome = create_safe_pie_chart(
                    outcome_counts,
                    names_col='outcome',
                    values_col='count',
                    title="Distribution of Patient Outcomes"
                )
                st.plotly_chart(fig_outcome, use_container_width=True)
            else:
                st.info("No dataset available")
            
        with col2:
            # Satisfaction Distribution
            st.subheader("⭐ Patient Satisfaction")
            if filtered_df is not None and 'satisfaction' in filtered_df.columns:
                satisfaction_counts = filtered_df['satisfaction'].value_counts().sort_index().reset_index()
                satisfaction_counts.columns = ['satisfaction_score', 'count']
                
                fig_satisfaction = create_safe_bar_chart(
                    satisfaction_counts,
                    x_col='satisfaction_score',
                    y_col='count',
                    title="Patient Satisfaction Scores",
                    color_col='count',
                    color_continuous_scale='Greens'
                )
                fig_satisfaction.update_layout(xaxis_title="Satisfaction Score", yaxis_title="Number of Patients")
                st.plotly_chart(fig_satisfaction, use_container_width=True)
            else:
                st.info("No dataset available")
        
        # Readmission Analysis
        st.subheader("🔄 Readmission Analysis by Condition")
        if filtered_df is not None and all(col in filtered_df.columns for col in ['condition', 'readmission']):
            try:
                readmission_analysis = filtered_df.groupby('condition').agg({
                    'readmission': lambda x: (x == 'Yes').mean() * 100,
                    'patient_id': 'count'
                }).round(2)
                readmission_analysis = readmission_analysis[readmission_analysis['patient_id'] > 3]
                readmission_analysis = readmission_analysis.sort_values('readmission', ascending=False).head(10).reset_index()
                readmission_analysis.columns = ['condition', 'readmission_rate', 'patient_count']
                
                fig_readmission = create_safe_bar_chart(
                    readmission_analysis,
                    x_col='condition',
                    y_col='readmission_rate',
                    title="Top 10 Conditions by Readmission Rate",
                    color_col='readmission_rate',
                    color_continuous_scale='Oranges'
                )
                fig_readmission.update_layout(xaxis_title="Medical Condition", yaxis_title="Readmission Rate (%)")
                st.plotly_chart(fig_readmission, use_container_width=True)
            except:
                st.info("No dataset available")
        else:
            st.info("No dataset available")
    
    # Data Table Section
    st.markdown("---")
    st.subheader("📋 Detailed Patient Data")
    
    # Show data summary
    if filtered_df is not None:
        st.write(f"**Displaying {len(filtered_df)} of {len(df)} patient records**")
    else:
        st.write("**No data available**")
    
    # Data table
    if filtered_df is not None and not filtered_df.empty:
        display_columns = []
        available_columns = ['patient_id', 'age', 'gender', 'condition', 'procedure', 'cost', 'length_of_stay', 'outcome', 'satisfaction', 'readmission']
        
        for col in available_columns:
            if col in filtered_df.columns:
                display_columns.append(col)
        
        if display_columns:
            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True,
                height=400
            )
            
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"hospital_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No dataset available")
    else:
        st.info("No dataset available")
    
    # Data Summary Statistics
    with st.expander("📊 View Detailed Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Data Summary**")
            if filtered_df is None or filtered_df.empty:
                st.info("No dataset available")
            else:
                try:
                    numerical_cols = [col for col in ['age', 'cost', 'length_of_stay', 'satisfaction'] if col in filtered_df.columns]
                    if numerical_cols:
                        numerical_data = filtered_df[numerical_cols].describe()
                        st.dataframe(numerical_data)
                    else:
                        st.info("No dataset available")
                except:
                    st.info("No dataset available")
        
        with col2:
            st.write("**Categorical Data Summary**")
            if filtered_df is None or filtered_df.empty:
                st.info("No dataset available")
            else:
                categorical_cols = ['gender', 'condition', 'procedure', 'outcome', 'readmission']
                for col in categorical_cols:
                    st.write(f"**{col.title()}:**")
                    try:
                        if col in filtered_df.columns and not filtered_df[col].isna().all():
                            counts = filtered_df[col].value_counts()
                            if len(counts) > 0:
                                counts_df = counts.reset_index()
                                counts_df.columns = ['Value', 'Count']
                                st.dataframe(counts_df)
                            else:
                                st.info("No dataset available")
                        else:
                            st.info("No dataset available")
                    except:
                        st.info("No dataset available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🏥 **Hospital Data Analysis Dashboard** | "
        "Treatment Pattern Insights | "
        "Cost Optimization | "
        "Patient Outcome Analysis"
    )

if __name__ == "__main__":
    main()

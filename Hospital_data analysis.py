import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import warnings

# Set page configuration
st.set_page_config(
    page_title="Hospital Data Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏥 Hospital Data Analysis Dashboard")
st.markdown("---")

# Sidebar for navigation and filters
st.sidebar.title("Navigation & Filters")

# Load REAL data from your CSV file
@st.cache_data
def load_real_data():
    try:
        # Try multiple locations to find your CSV file
        file_paths = [
            "hospital data analysis.csv",  # Same folder as script
            r"C:\Users\Zyon\Downloads\hospital data analysis.csv",  # Your downloads folder
            "hospital_data_analysis.csv"  # Alternative name
        ]
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                st.success(f"✅ Data loaded successfully from: {file_path}")
                return df
            except FileNotFoundError:
                continue
        
        # If file not found in any location, show upload option
        st.warning("📁 CSV file not found. Please upload your hospital data file.")
        uploaded_file = st.file_uploader("Choose your hospital data CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            return df
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Load data
df = load_real_data()

if df is None:
    st.error("❌ No data loaded. Please make sure your CSV file is available.")
    st.stop()

# Data preprocessing
try:
    # Clean column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Convert to proper data types
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    df['satisfaction'] = pd.to_numeric(df['satisfaction'], errors='coerce')
    df['length_of_stay'] = pd.to_numeric(df['length_of_stay'], errors='coerce')
    
    # Create a date column for time-based analysis (using patient_id as proxy)
    df['visit_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
    
    st.success("✅ Data preprocessing completed!")
    
except Exception as e:
    st.error(f"❌ Error in data preprocessing: {str(e)}")
    st.stop()

# Show dataset info in sidebar
st.sidebar.subheader("Dataset Information")
st.sidebar.write(f"Total Records: {len(df):,}")
st.sidebar.write(f"Data Columns: {', '.join(df.columns.tolist())}")

# Sidebar filters
st.sidebar.subheader("Data Filters")

# Condition filter
condition_options = ['All'] + list(df['condition'].unique())
selected_condition = st.sidebar.selectbox("Medical Condition", condition_options)

# Gender filter
gender_options = ['All'] + list(df['gender'].unique())
selected_gender = st.sidebar.selectbox("Gender", gender_options)

# Outcome filter
outcome_options = ['All'] + list(df['outcome'].unique())
selected_outcome = st.sidebar.selectbox("Outcome", outcome_options)

# Readmission filter
readmission_options = ['All'] + list(df['readmission'].unique())
selected_readmission = st.sidebar.selectbox("Readmission Status", readmission_options)

# Age range filter
min_age, max_age = int(df['age'].min()), int(df['age'].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# Cost range filter
min_cost, max_cost = int(df['cost'].min()), int(df['cost'].max())
cost_range = st.sidebar.slider("Cost Range (kes)", min_cost, max_cost, (min_cost, max_cost))

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

filtered_df = filtered_df[
    (filtered_df['age'] >= age_range[0]) &
    (filtered_df['age'] <= age_range[1]) &
    (filtered_df['cost'] >= cost_range[0]) &
    (filtered_df['cost'] <= cost_range[1])
]

# Main dashboard - KPI Metrics
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_patients = filtered_df['patient_id'].nunique()
    st.metric("Total Patients", f"{total_patients:,}")

with col2:
    avg_cost = filtered_df['cost'].mean()
    st.metric("Average Treatment Cost", f"kes{avg_cost:,.2f}")

with col3:
    avg_stay = filtered_df['length_of_stay'].mean()
    st.metric("Average Length of Stay", f"{avg_stay:.1f} days")

with col4:
    readmission_rate = (filtered_df['readmission'] == 'Yes').mean() * 100
    st.metric("Readmission Rate", f"{readmission_rate:.1f}%")

# Additional KPIs
col5, col6, col7, col8 = st.columns(4)

with col5:
    total_cost = filtered_df['cost'].sum()
    st.metric("Total Treatment Cost", f"kes{total_cost:,.0f}")

with col6:
    avg_satisfaction = filtered_df['satisfaction'].mean()
    st.metric("Average Satisfaction", f"{avg_satisfaction:.1f}/5")

with col7:
    recovery_rate = (filtered_df['outcome'] == 'Recovered').mean() * 100
    st.metric("Recovery Rate", f"{recovery_rate:.1f}%")

with col8:
    unique_conditions = filtered_df['condition'].nunique()
    st.metric("Unique Conditions", f"{unique_conditions}")

st.markdown("---")

# Visualization section
st.subheader("📈 Data Analysis & Visualizations")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Cost Analysis", "Patient Demographics", "Medical Conditions", "Treatment Patterns"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Treatment Cost Distribution**")
        fig = px.histogram(filtered_df, x='cost', nbins=20, 
                         title="Distribution of Treatment Costs",
                         color_discrete_sequence=['#1f77b4'])
        fig.update_layout(xaxis_title="Cost (kes)", yaxis_title="Number of Patients")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Cost by Medical Condition**")
        cost_by_condition = filtered_df.groupby('condition')['cost'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(cost_by_condition, 
                    x=cost_by_condition.values, 
                    y=cost_by_condition.index,
                    orientation='h',
                    title="Top 10 Conditions by Average Cost",
                    color=cost_by_condition.values,
                    color_continuous_scale='Viridis')
        fig.update_layout(xaxis_title="Average Cost (kes)", yaxis_title="Medical Condition")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Age Distribution**")
        fig = px.histogram(filtered_df, x='age', nbins=15, 
                         title="Patient Age Distribution",
                         color_discrete_sequence=['#2ecc71'])
        fig.update_layout(xaxis_title="Age", yaxis_title="Number of Patients")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Gender Distribution**")
        gender_counts = filtered_df['gender'].value_counts()
        fig = px.pie(values=gender_counts.values, 
                    names=gender_counts.index,
                    title="Gender Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Medical Conditions**")
        condition_counts = filtered_df['condition'].value_counts().head(10)
        fig = px.bar(condition_counts,
                    x=condition_counts.values,
                    y=condition_counts.index,
                    orientation='h',
                    title="Top 10 Medical Conditions",
                    color=condition_counts.values,
                    color_continuous_scale='Blues')
        fig.update_layout(xaxis_title="Number of Patients", yaxis_title="Medical Condition")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Patient Outcomes**")
        outcome_counts = filtered_df['outcome'].value_counts()
        fig = px.pie(values=outcome_counts.values,
                    names=outcome_counts.index,
                    title="Patient Outcomes Distribution",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Length of Stay vs Treatment Cost**")
        fig = px.scatter(filtered_df,
                       x='length_of_stay',
                       y='cost',
                       color='condition',
                       size='age',
                       hover_data=['gender', 'procedure'],
                       title="Relationship between Length of Stay and Treatment Cost")
        fig.update_layout(xaxis_title="Length of Stay (Days)", yaxis_title="Treatment Cost ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Readmission Analysis**")
        readmission_by_condition = filtered_df.groupby('condition').agg({
            'readmission': lambda x: (x == 'Yes').mean() * 100,
            'patient_id': 'count'
        }).round(2)
        readmission_by_condition = readmission_by_condition[readmission_by_condition['patient_id'] > 5]
        readmission_by_condition = readmission_by_condition.sort_values('readmission', ascending=False).head(10)
        
        fig = px.bar(readmission_by_condition,
                    x=readmission_by_condition.index,
                    y='readmission',
                    title="Readmission Rates by Condition (Top 10)",
                    color='readmission',
                    color_continuous_scale='Reds')
        fig.update_layout(xaxis_title="Medical Condition", yaxis_title="Readmission Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

# Data table section
st.markdown("---")
st.subheader("📋 Patient Data Records")

# Show filtered data
st.write(f"Showing {len(filtered_df)} of {len(df)} patient records")
st.dataframe(filtered_df, use_container_width=True)

# Download button for filtered data
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name=f"hospital_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("🏥 **Hospital Data Analysis Dashboard** - Analyzing real patient treatment patterns and outcomes")
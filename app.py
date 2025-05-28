import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Predictor v2.0",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #4caf50;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_components():
    """Load all model components with caching for better performance"""
    try:
        # Load from models directory
        model = joblib.load('models/energy_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le_building = joblib.load('models/le_building.pkl')
        le_day = joblib.load('models/le_day.pkl')
        
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
            
        return model, scaler, le_building, le_day, model_info
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        st.error("Please ensure all model files are in the 'models/' directory")
        return None, None, None, None, None

def prepare_features(building_type, square_footage, num_occupants, 
                    appliances_used, avg_temperature, day_of_week,
                    scaler, le_building, le_day):
    """Prepare features for prediction"""
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Building Type': [building_type],
            'Square Footage': [square_footage],
            'Number of Occupants': [num_occupants],
            'Appliances Used': [appliances_used],
            'Average Temperature': [avg_temperature],
            'Day of Week': [day_of_week]
        })
        
        # Encode categorical variables
        input_data['Building Type Encoded'] = le_building.transform(input_data['Building Type'])
        input_data['Day of Week Encoded'] = le_day.transform(input_data['Day of Week'])
        
        # Create additional features
        input_data['Occupancy_Density'] = input_data['Number of Occupants'] / input_data['Square Footage']
        input_data['Appliances_per_Occupant'] = input_data['Appliances Used'] / (input_data['Number of Occupants'] + 1)
        
        # Size category
        if square_footage <= 15000:
            size_cat = 0  # Small
        elif square_footage <= 30000:
            size_cat = 1  # Medium
        else:
            size_cat = 2  # Large
        
        input_data['Size_Category_Encoded'] = size_cat
        
        # Select and order features
        feature_columns = ['Square Footage', 'Number of Occupants', 'Appliances Used', 
                          'Average Temperature', 'Building Type Encoded', 'Day of Week Encoded',
                          'Occupancy_Density', 'Appliances_per_Occupant', 'Size_Category_Encoded']
        
        X = input_data[feature_columns]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        return X_scaled, input_data
    except Exception as e:
        st.error(f"Error in feature preparation: {e}")
        return None, None

def create_comparison_chart(prediction, building_type):
    """Create comparison visualization"""
    building_averages = {
        'Residential': 2500,
        'Commercial': 4200,
        'Industrial': 6800
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Your Building', f'Average {building_type}'],
        y=[prediction, building_averages[building_type]],
        marker_color=['#FF6B6B', '#4ECDC4'],
        text=[f'{prediction:.0f}', f'{building_averages[building_type]:.0f}'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Energy Consumption Comparison",
        yaxis_title="Energy Consumption (units)",
        showlegend=False,
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_efficiency_gauge(efficiency_score):
    """Create efficiency gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = efficiency_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Efficiency Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Load model components
    model, scaler, le_building, le_day, model_info = load_model_components()
    
    if model is None:
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🏢 Energy Consumption Predictor v2.0</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced ML-powered building energy consumption prediction")
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        if model_info:
            st.markdown(f"**Model:** {model_info['model_name']}")
            st.markdown(f"**Version:** {model_info.get('project_version', '1.0')}")
            st.markdown(f"**Training Date:** {model_info['training_date']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{model_info['r2_score']:.3f}")
                st.metric("MAE", f"{model_info['mae']:.2f}")
            with col2:
                st.metric("RMSE", f"{model_info['rmse']:.2f}")
        
        st.markdown("---")
        st.markdown("**Features Used:**")
        if model_info and 'features' in model_info:
            for feature in model_info['features']:
                st.markdown(f"• {feature}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔧 Building Configuration")
        
        # Input form
        with st.form("prediction_form"):
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                building_type = st.selectbox(
                    "Building Type",
                    options=['Residential', 'Commercial', 'Industrial'],
                    help="Select the type of building"
                )
                
                square_footage = st.number_input(
                    "Square Footage",
                    min_value=500,
                    max_value=100000,
                    value=2500,
                    step=100,
                    help="Total building area in square feet"
                )
                
                num_occupants = st.number_input(
                    "Number of Occupants",
                    min_value=1,
                    max_value=500,
                    value=10,
                    help="Average number of people in the building"
                )
            
            with col1_2:
                appliances_used = st.number_input(
                    "Number of Appliances",
                    min_value=1,
                    max_value=100,
                    value=15,
                    help="Total number of electrical appliances"
                )
                
                avg_temperature = st.slider(
                    "Average Temperature (°C)",
                    min_value=-10,
                    max_value=40,
                    value=20,
                    help="Average ambient temperature"
                )
                
                day_of_week = st.selectbox(
                    "Day Type",
                    options=['Weekday', 'Weekend'],
                    help="Type of day for prediction"
                )
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Energy Consumption", type="primary")
        
        if submitted:
            with st.spinner("Calculating prediction..."):
                # Prepare features
                X_scaled, input_data = prepare_features(
                    building_type, square_footage, num_occupants,
                    appliances_used, avg_temperature, day_of_week,
                    scaler, le_building, le_day
                )
                
                if X_scaled is not None:
                    # Make prediction
                    prediction = model.predict(X_scaled)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.header("📈 Prediction Results")
                    
                    # Main prediction result
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2>Predicted Energy Consumption</h2>
                        <h1 style="color: #4caf50; margin: 0;">{prediction:.2f} units</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Efficiency metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    energy_per_sqft = prediction / square_footage
                    energy_per_occupant = prediction / num_occupants
                    efficiency_score = max(0, min(100, 100 - (prediction / 100)))  # Simple efficiency calculation
                    
                    with col_m1:
                        st.metric("Energy per Sq Ft", f"{energy_per_sqft:.4f}")
                    with col_m2:
                        st.metric("Energy per Occupant", f"{energy_per_occupant:.2f}")
                    with col_m3:
                        st.metric("Efficiency Score", f"{efficiency_score:.1f}/100")
                    
                    # Visualizations
                    st.subheader("📊 Analysis & Insights")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Comparison chart
                        comparison_fig = create_comparison_chart(prediction, building_type)
                        st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    with viz_col2:
                        # Efficiency gauge
                        gauge_fig = create_efficiency_gauge(efficiency_score)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("🎯 Feature Importance")
                        feature_names = ['Square Footage', 'Number of Occupants', 'Appliances Used', 
                                       'Average Temperature', 'Building Type', 'Day of Week',
                                       'Occupancy Density', 'Appliances per Occupant', 'Size Category']
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=True)
                        
                        fig_importance = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title="Feature Importance for Your Prediction"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.header("💡 Quick Predictions")
        
        # Example predictions
        examples = [
            ("Small Office", "Commercial", 1500, 8, 12, 22, "Weekday"),
            ("Family Home", "Residential", 2000, 4, 15, 20, "Weekend"),
            ("Large Factory", "Industrial", 50000, 200, 80, 18, "Weekday"),
            ("Apartment", "Residential", 800, 2, 8, 24, "Weekday"),
            ("Retail Store", "Commercial", 5000, 15, 25, 21, "Weekend")
        ]
        
        for name, b_type, sq_ft, occupants, appliances, temp, day in examples:
            if st.button(f"🏠 {name}", key=name, use_container_width=True):
                X_scaled, _ = prepare_features(
                    b_type, sq_ft, occupants, appliances, temp, day,
                    scaler, le_building, le_day
                )
                if X_scaled is not None:
                    pred = model.predict(X_scaled)[0]
                    st.success(f"**{name}**: {pred:.0f} units")
        
        # st.markdown("---")
        # st.header("📋 Usage Tips")
        # st.markdown("""
        # **For better predictions:**
        # - Ensure accurate building measurements
        # - Consider seasonal temperature variations
        # - Account for actual occupancy patterns
        # - Include all major appliances
        
        # **Interpretation:**
        # - Higher efficiency scores are better
        # - Compare with similar building types
        # - Use for energy planning and budgeting
        # """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Energy Consumption Predictor v2.0</strong> | 
        Powered by Machine Learning | 
        Built with Streamlit</p>
        <p><em>Predictions are estimates based on historical data and should be used as guidance.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

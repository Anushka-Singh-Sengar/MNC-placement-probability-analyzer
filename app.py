import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="MNC Placement Probability Analyzer",
    page_icon="📊",
    layout="centered"
)

# Title and description
st.title("🎯 MNC Placement Probability Analyzer")
st.markdown("### Predict your readiness for top MNC placements")

# Company selection
companies = {
    'Google': ['CGPA', 'Total Problems Solved', 'LeetCode Solved'],
    'Microsoft': ['Technical Projects', 'Internships', 'Certifications', 'Total Problems Solved', 'Total Skills'],
    'Amazon': ['Technical Projects', 'Internships', 'Certifications', 'LeetCode Solved', 'Teamwork Experience'],
    'Infosys': ['Internships', 'Technical Projects', 'Certifications', 'LeetCode Solved', 'Teamwork Experience']
}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "How It Works"])

if page == "Home":
    # Company selection
    selected_company = st.selectbox(
        "Select a company to check your placement probability:",
        list(companies.keys())
    )

    # Input fields based on company
    st.subheader(f"Enter your details for {selected_company}")
    
    # Create input fields for each feature
    input_data = {}
    for feature in companies[selected_company]:
        # Set default values and min/max based on feature
        min_val = 0
        max_val = 10
        step = 0.1 if feature == 'CGPA' else 1
        value = 0.0 if feature == 'CGPA' else 0
        
        if feature == 'CGPA':
            min_val = 0.0
            max_val = 10.0
            value = 8.0  # Default CGPA
        elif feature in ['Total Problems Solved', 'LeetCode Solved']:
            max_val = 1000
            value = 100
        elif feature in ['Technical Projects', 'Internships', 'Certifications', 'Total Skills', 'Teamwork Experience']:
            max_val = 50
            value = 5
        
        input_data[feature] = st.slider(
            label=f"{feature}:",
            min_value=min_val,
            max_value=max_val,
            value=value,
            step=step,
            help=f"Enter your {feature}"
        )

    # Predict button
    if st.button("Predict My Readiness"):
        try:
            # Load the model
            model_path = Path(f"models/{selected_company.lower()}.joblib")
            if not model_path.exists():
                st.error(f"Model for {selected_company} not found. Please train the model first.")
            else:
                model_data = joblib.load(model_path)
                model = model_data['model']
                
                # Prepare input data
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                probability = max(0.0, min(1.0, float(prediction))) * 100
                
                # Display result
                st.success("### 🎯 Your Results")
                
                # Progress bar for visualization
                st.metric(
                    label=f"Your {selected_company} Readiness Score:",
                    value=f"{probability:.1f}%"
                )
                st.progress(int(probability) / 100)
                
                # Suggestions based on score
                st.subheader("📝 Suggestions")
                if probability < 30:
                    st.warning("### Areas to Improve:")
                    st.markdown("""
                    - Focus on core DSA and problem-solving
                    - Work on personal projects
                    - Consider getting certifications
                    - Practice coding on platforms like LeetCode
                    """)
                elif probability < 60:
                    st.info("### You're on the right track!")
                    st.markdown("""
                    - Practice more coding problems
                    - Consider an internship
                    - Work on team projects
                    - Improve your problem-solving speed
                    """)
                else:
                    st.success("### Great job! Keep it up!")
                    st.markdown("""
                    - Prepare for behavioral interviews
                    - Practice system design
                    - Keep solving challenging problems
                    - Network with professionals
                    """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please make sure you have trained the models first by running 'python -m src.mnc_probability_analyzer.train'")

else:  # How It Works page
    st.header("How It Works")
    st.markdown("""
    ## 📊 MNC Placement Probability Analyzer
    
    This tool helps you predict your readiness for placements in top MNCs based on various factors:
    - Academic performance (CGPA)
    - Coding practice (LeetCode, problem-solving)
    - Projects and internships
    - Certifications
    - Soft skills
    
    ### 🚀 Getting Started
    1. Select a company from the dropdown
    2. Enter your details in the input fields
    3. Click "Predict My Readiness" to see your score
    
    ### ⚙️ Technical Details
    - Uses Linear Regression models trained on historical data
    - Each company has a custom model with relevant features
    - Models are saved in the `models/` directory
    
    ### 🔧 Setup
    ```bash
    # Install dependencies
    pip install -r requirements.txt
    
    # Train the models (required before first use)
    python -m src.mnc_probability_analyzer.train
    
    # Run the web app
    streamlit run app.py
    ```
    """)

# Footer
st.markdown("---")
st.markdown("### 📌 Note")
st.markdown("""
- This is a predictive model and should be used for guidance only.
- Actual placement outcomes may vary based on many factors.
- The model is trained on limited data and should be regularly updated.
""")

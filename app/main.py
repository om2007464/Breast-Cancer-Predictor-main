import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Function to load and clean the data
def get_clean_data():
    data = pd.read_csv("data/data.csv")  # Load data
    data = data.drop(['Unnamed: 32', 'id'], axis=1)  # Drop unnecessary columns
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Map diagnosis to binary values
    return data

# Function to add sidebar elements
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")  # Sidebar header
    
    data = get_clean_data()  # Get cleaned data
    
    # Define slider labels and corresponding keys
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}  # Dictionary to store slider values

    # Add sliders to the sidebar
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

# Function to scale the input values
def get_scaled_values(input_dict):
    data = get_clean_data()  # Get cleaned data
    X = data.drop(['diagnosis'], axis=1)  # Drop the target variable
    
    scaled_dict = {}  # Dictionary to store scaled values
    
    # Scale each input value
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

# Function to create a radar chart
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)  # Scale input data
    
    # Define categories for radar chart
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    # Add traces for mean, standard error, and worst values
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    
    return fig

# Function to add prediction results
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))  # Load model
    scaler = pickle.load(open("model/scaler.pkl", "rb"))  # Load scaler
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)  # Convert input to array
    
    input_array_scaled = scaler.transform(input_array)  # Scale input array
    
    prediction = model.predict(input_array_scaled)  # Predict diagnosis
    
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    
    # Display prediction result
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    
    # Display prediction probabilities
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

# Main function to run the app
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("CSS/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()  # Get input data from sidebar
    
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app determines whether a breast mass is benign or malignant based on measurements received from your cytology lab. You can also manually adjust these measurements using the sliders in the sidebar.")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)  # Generate radar chart
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)  # Add prediction results

if __name__ == '__main__':
    main()

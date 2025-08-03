import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from interpret import show
from interpret.provider import InlineProvider
from alibi.explainers import AnchorTabular
import warnings

warnings.filterwarnings('ignore')

# TensorFlow configuration
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

tf.get_logger().setLevel(40)

# Page configuration
st.set_page_config(
    page_title="ASD Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 1rem 0;
    }

    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }

    .success-box {
        background: linear-gradient(135deg, #d4f1d4 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        margin: 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Set font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# Predictor classes (consistent with save_models)
class APLRPredictor:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x_df = pd.DataFrame(x, columns=self.feature_names)
            predictions = self.model.predict(x_df)
            return np.array([int(p) for p in predictions])
        return np.array([0])


class RFPredictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            pred_labels = self.model.predict(self.scaler.transform(x))
            pred_proba = np.zeros((len(pred_labels), 2))
            for i, label in enumerate(pred_labels):
                pred_proba[i, int(label)] = 1.0
            return pred_proba
        return np.array([[1.0, 0.0]])


class EnhancedModelLoader:
    """Load all saved models and explanation components"""

    def __init__(self):
        self.aplr_models = {}
        self.rf_models = {}
        self.scalers = {}
        self.anchor_data = {}
        self.global_explanations = {}
        self.local_explanations = {}
        self.data_info = {}
        self.training_data = {}

    @st.cache_resource
    def load_all_models(_self):
        """Load all saved models and results"""
        try:
            for region in ["L34", "L5S1"]:
                # Load APLR models
                _self.aplr_models[region] = joblib.load(f"models/{region}_aplr_model.joblib")

                # Load Random Forest models and scalers
                _self.rf_models[region] = joblib.load(f"models/{region}_rf_model.joblib")
                _self.scalers[region] = joblib.load(f"models/{region}_scaler.joblib")

                # Load anchor explainer data
                with open(f"models/{region}_anchor_data.pkl", 'rb') as f:
                    _self.anchor_data[region] = pickle.load(f)

                # Load explanation results
                with open(f"models/{region}_global_explanation.pkl", 'rb') as f:
                    _self.global_explanations[region] = pickle.load(f)

                with open(f"models/{region}_local_explanations.pkl", 'rb') as f:
                    _self.local_explanations[region] = pickle.load(f)

                # Load data info
                with open(f"models/{region}_data_info.pkl", 'rb') as f:
                    _self.data_info[region] = pickle.load(f)

                # Load training data
                with open(f"models/{region}_training_data.pkl", 'rb') as f:
                    _self.training_data[region] = pickle.load(f)

            return True
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return False

    def create_anchor_explainer(self, region):
        """Dynamically create anchor explainer"""
        try:
            anchor_data = self.anchor_data[region]
            aplr_model = self.aplr_models[region]

            # Create predictor
            predictor = APLRPredictor(aplr_model, anchor_data['feature_names'])

            # Create anchor explainer
            anchor_explainer = AnchorTabular(
                predictor=predictor,
                feature_names=anchor_data['feature_names'],
                categorical_names=anchor_data['categorical_features'] if anchor_data['categorical_features'] else None
            )

            # Refit
            anchor_explainer.fit(anchor_data['training_data'], disc_perc=anchor_data['disc_perc'])

            return anchor_explainer
        except Exception as e:
            st.error(f"Anchor explainer creation failed: {e}")
            return None


# Initialize model loader
@st.cache_resource
def get_model_loader():
    return EnhancedModelLoader()


def is_parameter_in_normal_range(feature, value):
    """Check if parameter is in normal range"""
    normal_ranges = {
        'Cage height': (10, 14),
        'Operative time ': (60, 180),
        'Blood loss': (50, 200)
    }

    if feature in normal_ranges:
        min_val, max_val = normal_ranges[feature]
        return min_val <= value <= max_val

    return True


def attempt_counterfactual_analysis(loader, segment, sample_array, prediction, feature_names):
    """Attempt counterfactual analysis"""
    try:
        # Target class is opposite of current prediction
        target_class = 1 - prediction

        # Load necessary components
        rf_model = loader.rf_models[segment]
        scaler = loader.scalers[segment]
        training_data = loader.training_data[segment]

        # Build counterfactual predictor
        rf_predictor = RFPredictor(rf_model, scaler)

        # Try simple grid search for counterfactual analysis
        interventional_features = ['Cage height', 'Operative time ', 'Blood loss']
        interventional_indices = []

        for i, feature in enumerate(feature_names):
            if feature in interventional_features:
                interventional_indices.append(i)

        if not interventional_indices:
            return {
                'success': False,
                'reason': 'No available intervention features',
                'changes': [],
                'target_class': target_class
            }

        # Simple parameter adjustment strategy
        original_sample = sample_array.copy()
        best_changes = None

        # Define parameter adjustment ranges
        adjustment_ranges = {
            'Cage height': [-2, -1, 1, 2],  # ±1-2mm
            'Operative time ': [-30, -15, 15, 30],  # ±15-30 minutes
            'Blood loss': [-50, -25, 25, 50]  # ±25-50ml
        }

        # Try different parameter combinations
        for cage_adj in adjustment_ranges.get('Cage height', [0]):
            for time_adj in adjustment_ranges.get('Operative time ', [0]):
                for blood_adj in adjustment_ranges.get('Blood loss', [0]):

                    # Create adjusted sample
                    adjusted_sample = original_sample.copy()
                    changes = []

                    # Apply adjustments
                    for i, feature in enumerate(feature_names):
                        if feature == 'Cage height':
                            new_value = max(8, min(16, original_sample[i] + cage_adj))
                            if abs(new_value - original_sample[i]) > 0.1:
                                adjusted_sample[i] = new_value
                                changes.append({
                                    'Parameter': feature,
                                    'Original Value': original_sample[i],
                                    'Adjusted Value': new_value,
                                    'Change': new_value - original_sample[i]
                                })
                        elif feature == 'Operative time ':
                            new_value = max(40, min(300, original_sample[i] + time_adj))
                            if abs(new_value - original_sample[i]) > 1:
                                adjusted_sample[i] = new_value
                                changes.append({
                                    'Parameter': feature,
                                    'Original Value': original_sample[i],
                                    'Adjusted Value': new_value,
                                    'Change': new_value - original_sample[i]
                                })
                        elif feature == 'Blood loss':
                            new_value = max(20, min(400, original_sample[i] + blood_adj))
                            if abs(new_value - original_sample[i]) > 1:
                                adjusted_sample[i] = new_value
                                changes.append({
                                    'Parameter': feature,
                                    'Original Value': original_sample[i],
                                    'Adjusted Value': new_value,
                                    'Change': new_value - original_sample[i]
                                })

                    if changes:
                        # Test adjusted prediction results
                        try:
                            pred_proba = rf_predictor(adjusted_sample.reshape(1, -1))
                            new_prediction = np.argmax(pred_proba[0])

                            if new_prediction == target_class:
                                return {
                                    'success': True,
                                    'reason': 'Successfully found counterfactual explanation',
                                    'changes': changes,
                                    'target_class': target_class,
                                    'new_probability': pred_proba[0]
                                }
                        except:
                            continue

        return {
            'success': False,
            'reason': 'No combination found within reasonable parameter adjustment range',
            'changes': [],
            'target_class': target_class
        }

    except Exception as e:
        return {
            'success': False,
            'reason': f'Counterfactual analysis error: {str(e)}',
            'changes': [],
            'target_class': 1 - prediction if 'prediction' in locals() else 0
        }


def generate_global_explanation_plots(loader, region):
    """
    Generate global explanation plots - following attachment implementation
    Use unique_term_affiliations instead of feature_names
    """
    try:
        aplr_model = loader.aplr_models[region]

        # Use unique_term_affiliations to get feature importance
        feature_importance = aplr_model.get_feature_importance()
        unique_affiliations = aplr_model.get_unique_term_affiliations()

        # Ensure array lengths match
        if len(feature_importance) != len(unique_affiliations):
            min_len = min(len(feature_importance), len(unique_affiliations))
            feature_importance = feature_importance[:min_len]
            unique_affiliations = unique_affiliations[:min_len]

        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': unique_affiliations,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)

        # Show only top 15 most important features
        top_features = importance_df.tail(15)

        # Create feature importance plot
        fig_importance = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'{region} Global Term/Feature Importances (Top 15)',
            labels={'Importance': 'Standard deviation of contribution to linear predictor', 'Feature': 'Feature Name'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig_importance.update_layout(
            height=600,
            title_font_size=16,
            font_color='#333333',
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_importance.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
        fig_importance.update_yaxes(gridcolor='lightgray', gridwidth=0.5)

        return fig_importance, importance_df

    except Exception as e:
        st.error(f"Global explanation plot generation failed: {e}")
        return None, None


def generate_local_explanation_plots(loader, region, sample_data, feature_names):
    """
    Generate local explanation plots - following attachment implementation
    Use calculate_local_feature_contribution method
    """
    try:
        aplr_model = loader.aplr_models[region]

        # Calculate local feature contribution
        local_contributions = aplr_model.calculate_local_feature_contribution(sample_data.values)
        unique_affiliations = aplr_model.get_unique_term_affiliations()

        if len(local_contributions) > 0:
            # Get first sample's contribution
            contributions = local_contributions[0]

            # Ensure array lengths match
            if len(contributions) != len(unique_affiliations):
                min_len = min(len(contributions), len(unique_affiliations))
                contributions = contributions[:min_len]
                unique_affiliations = unique_affiliations[:min_len]

            # Create local explanation dataframe
            local_df = pd.DataFrame({
                'Feature': unique_affiliations,
                'Contribution': contributions
            })

            # Sort by absolute value, show features with highest impact
            local_df['Abs_Contribution'] = local_df['Contribution'].abs()
            local_df = local_df.sort_values('Abs_Contribution', ascending=True)

            # Show only top 15 features with highest impact
            top_local = local_df.tail(15)

            # Create local explanation plot
            colors = ['#d62728' if x < 0 else '#1f77b4' for x in top_local['Contribution']]

            fig_local = go.Figure(data=[
                go.Bar(
                    x=top_local['Contribution'],
                    y=top_local['Feature'],
                    orientation='h',
                    marker_color=colors,
                    text=[f'{x:.3f}' for x in top_local['Contribution']],
                    textposition='auto'
                )
            ])

            fig_local.update_layout(
                title=f'{region} Local Feature Contributions (Top 15)',
                xaxis_title='Feature Contribution Value',
                yaxis_title='Feature Name',
                height=600,
                title_font_size=16,
                font_color='#333333',
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig_local.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
            fig_local.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
            fig_local.update_yaxes(gridcolor='lightgray', gridwidth=0.5)

            return fig_local, local_df
        else:
            return None, None

    except Exception as e:
        st.error(f"Local explanation plot generation failed: {e}")
        return None, None


def show_aplr_model_summary(loader, region):
    """Display APLR model summary information"""
    try:
        aplr_model = loader.aplr_models[region]

        st.markdown('<div class="section-header">🤖 APLR Model Details</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            try:
                cv_error = aplr_model.get_cv_error()
                st.markdown(f'<div class="metric-card"><h3>{cv_error:.4f}</h3><p>CV Error</p></div>',
                            unsafe_allow_html=True)
            except:
                st.markdown(f'<div class="metric-card"><h3>N/A</h3><p>CV Error</p></div>', unsafe_allow_html=True)

        with col2:
            try:
                feature_importance = aplr_model.get_feature_importance()
                st.markdown(f'<div class="metric-card"><h3>{len(feature_importance)}</h3><p>Features</p></div>',
                            unsafe_allow_html=True)
            except:
                st.markdown(f'<div class="metric-card"><h3>N/A</h3><p>Features</p></div>', unsafe_allow_html=True)

        with col3:
            try:
                unique_affiliations = aplr_model.get_unique_term_affiliations()
                st.markdown(f'<div class="metric-card"><h3>{len(unique_affiliations)}</h3><p>Model Terms</p></div>',
                            unsafe_allow_html=True)
            except:
                st.markdown(f'<div class="metric-card"><h3>N/A</h3><p>Model Terms</p></div>', unsafe_allow_html=True)

        # Display model term information
        with st.expander("📊 View Detailed Model Terms"):
            try:
                feature_importance = aplr_model.get_feature_importance()
                unique_affiliations = aplr_model.get_unique_term_affiliations()

                # Ensure lengths match
                if len(feature_importance) == len(unique_affiliations):
                    importance_df = pd.DataFrame({
                        'Feature Name': unique_affiliations,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=False)

                    st.dataframe(importance_df, use_container_width=True)
                else:
                    st.info("Feature importance data length mismatch, unable to display details")
            except Exception as e:
                st.info(f"Model term details unavailable: {e}")

    except Exception as e:
        st.warning(f"Model summary generation failed: {e}")


def show_homepage():
    """Display homepage"""
    st.markdown('<div class="main-header">🏥 Adjacent Segment Disease (ASD) Prediction System</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Create attractive info boxes
    st.markdown("""
    <div class="info-box">
        <h2>🎯 Welcome to the ASD Prediction System</h2>
        <p>This is an explainable AI-based adjacent segment disease prediction application, developed based on research from <strong>Beijing Chaoyang Hospital, Capital Medical University</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📈 Research Data
        - **L3-4 Segment**: 421 patients, 101 ASD cases (24.0%)
        - **L5-S1 Segment**: 421 patients, 84 ASD cases (19.9%)
        - **Follow-up**: Minimum 4 years
        - **Research Institution**: Beijing Chaoyang Hospital, Capital Medical University

        ### 🔬 Technical Features
        - **APLR Model**: Automatic Piecewise Linear Regression with transparent predictions
        - **Anchor Explanations**: Identify key decision rules
        - **Counterfactual Analysis**: Provide intervention recommendations
        """)

    with col2:
        st.markdown("""
        ### 📋 Usage Instructions
        1. **Research Results**: View global and local explanation results
        2. **Smart Prediction**: Input patient data for ASD risk prediction with explanations

        ### 🚀 Getting Started
        Please select a function from the sidebar menu to begin.
        """)

    # Disclaimer section
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ IMPORTANT DISCLAIMER</h3>
        <p><strong>This system is for academic research and educational purposes only and cannot replace professional medical judgment.</strong></p>
        <ul>
            <li>This prediction system is developed based on specific research datasets and may not be applicable to all patient populations</li>
            <li>Prediction results are for reference only and should not be the sole basis for clinical decisions</li>
            <li>All medical decisions should be made by qualified medical professionals</li>
            <li>Users are responsible for any consequences arising from the use of this system</li>
            <li>Developers are not responsible for the accuracy of predictions or any resulting losses</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def show_dataset_overview(loader):
    """Display dataset overview"""
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for i, (region, col) in enumerate([("L34", col1), ("L5S1", col2)]):
        with col:
            region_name = "L3-4" if region == "L34" else "L5-S1"
            st.subheader(f"{region_name} Dataset")

            data_info = loader.data_info[region]
            total_patients = data_info['data_shape'][0]
            asd_patients = data_info['asd_distribution'][1]
            no_asd_patients = data_info['asd_distribution'][0]

            # Create metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Patients", total_patients)
                st.metric("ASD Cases", f"{asd_patients} ({asd_patients / total_patients * 100:.1f}%)")
            with col_b:
                st.metric("Non-ASD Cases", f"{no_asd_patients} ({no_asd_patients / total_patients * 100:.1f}%)")

            # Distribution pie chart
            fig = px.pie(
                values=[asd_patients, no_asd_patients],
                names=['ASD', 'Non-ASD'],
                title=f"{region_name} ASD Distribution",
                color_discrete_sequence=['#e74c3c', '#2ecc71']
            )
            fig.update_layout(
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)


def show_research_results(loader):
    """Display research results"""
    st.markdown('<div class="main-header">🔬 Research Results</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🌍 Global Explanations", "🔍 Local Explanations"])

    with tab1:
        show_dataset_overview(loader)

    with tab2:
        st.markdown('<div class="section-header">🌍 APLR Model Global Explanations</div>', unsafe_allow_html=True)
        region_choice = st.selectbox("Select Dataset", ["L34", "L5S1"],
                                     format_func=lambda x: "L3-4" if x == "L34" else "L5-S1")

        data_info = loader.data_info[region_choice]
        region_name = "L3-4" if region_choice == "L34" else "L5-S1"

        # Display model basic information
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{data_info["data_shape"][0]}</h3><p>Total Samples</p></div>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{data_info["data_shape"][1] - 1}</h3><p>Features</p></div>',
                        unsafe_allow_html=True)

        # Generate and display global explanation plots
        st.subheader("🎯 Feature Importance Analysis")
        fig_importance, importance_df = generate_global_explanation_plots(loader, region_choice)

        if fig_importance:
            st.plotly_chart(fig_importance, use_container_width=True)

            # Display detailed feature importance table
            with st.expander("📋 View Detailed Feature Importance Data"):
                st.dataframe(importance_df.sort_values('Importance', ascending=False), use_container_width=True)

        # Display key findings summary
        st.subheader("🔍 Key Findings Summary")
        if region_choice == "L34":
            st.markdown("""
            <div class="success-box">
                <h4>L3-4 Segment Key Findings</h4>
                <ul>
                    <li><strong>Most Important Feature</strong>: L3-4 and L5-S1 EBQ interaction</li>
                    <li><strong>Key Risk Factors</strong>: Endplate bone quality, disc degeneration</li>
                    <li><strong>Feature Pattern</strong>: Complex nonlinear interaction between age and EBQ</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                <h4>L5-S1 Segment Key Findings</h4>
                <ul>
                    <li><strong>Most Important Feature</strong>: Facet joint osteoarthritis</li>
                    <li><strong>Key Risk Factors</strong>: Facet joint degeneration, bone quality</li>
                    <li><strong>Feature Pattern</strong>: Multi-factorial comprehensive interaction</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">🔍 Representative Sample Local Explanations</div>',
                    unsafe_allow_html=True)
        region_choice = st.selectbox("Select Dataset", ["L34", "L5S1"],
                                     format_func=lambda x: "L3-4" if x == "L34" else "L5-S1", key="local_region")

        local_exps = loader.local_explanations[region_choice]
        feature_names = loader.data_info[region_choice]['feature_names']

        # Sample selection
        sample_options = []
        for exp in local_exps:
            label = "ASD" if exp['true_label'] == 1 else "Non-ASD"
            sample_options.append(f"Sample {exp['index']} ({label})")

        selected_sample = st.selectbox("Select Sample to View", sample_options)
        sample_idx = int(selected_sample.split()[1])

        # Find corresponding explanation
        selected_exp = None
        for exp in local_exps:
            if exp['index'] == sample_idx:
                selected_exp = exp
                break

        if selected_exp:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("""
                <div class="info-box">
                    <h4>📋 Sample Information</h4>
                </div>
                """, unsafe_allow_html=True)

                st.write(f"**Sample Index**: {selected_exp['index']}")
                st.write(f"**True Label**: {'ASD' if selected_exp['true_label'] == 1 else 'Non-ASD'}")

                # Display key features
                st.subheader("🏷️ Key Feature Values")
                key_features = ['Gender', 'Age', 'BMI', 'Hypertension', 'Diabetes', 'L3-4 EBQ', 'L5-S1 EBQ']
                features = selected_exp['features']

                for feature in key_features:
                    if feature in features:
                        value = features[feature]
                        if feature == 'Gender':
                            display_value = "Male" if value == 1 else "Female"
                        elif feature in ['Hypertension', 'Diabetes']:
                            display_value = "Yes" if value == 1 else "No"
                        else:
                            display_value = f"{value}"
                        st.write(f"**{feature}**: {display_value}")

            with col2:
                st.subheader("📊 Local Feature Contribution Analysis")

                # Build sample data
                sample_data = pd.DataFrame([list(features.values())], columns=list(features.keys()))
                # Ensure feature order matches training
                sample_data = sample_data.reindex(columns=feature_names, fill_value=0)

                # Generate local explanation plots
                fig_local, local_df = generate_local_explanation_plots(loader, region_choice, sample_data,
                                                                       feature_names)

                if fig_local:
                    st.plotly_chart(fig_local, use_container_width=True)

                    with st.expander("📋 View Detailed Local Contribution Data"):
                        st.dataframe(local_df.sort_values('Abs_Contribution', ascending=False),
                                     use_container_width=True)
                else:
                    st.warning("Unable to generate local explanation plot")


def show_typical_anchor_rules(segment, prediction):
    """Display typical anchor rules based on research"""
    if segment == "L34":
        if prediction == 1:  # High risk
            st.markdown("""
            <div class="info-box">
                <h4>📋 Research-Based Typical High-Risk Rules:</h4>
                <ul>
                    <li>L3-4 EBQ > 3.6 AND L5-S1 EBQ ≤ 3.5</li>
                    <li>L3-4 Pfirrmann grade ≥ 4 AND foraminal stenosis present</li>
                    <li>Age > 65 years AND high endplate bone quality score</li>
                </ul>
                <p>💡 These combination conditions are closely associated with high ASD risk in our research</p>
            </div>
            """, unsafe_allow_html=True)
        else:  # Low risk
            st.markdown("""
            <div class="success-box">
                <h4>📋 Research-Based Typical Low-Risk Features:</h4>
                <ul>
                    <li>L3-4 EBQ ≤ 2.5 AND no significant disc degeneration</li>
                    <li>Age < 60 years AND good bone quality</li>
                    <li>No foraminal stenosis AND normal facet joint function</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:  # L5S1
        if prediction == 1:  # High risk
            st.markdown("""
            <div class="info-box">
                <h4>📋 Research-Based Typical High-Risk Rules:</h4>
                <ul>
                    <li>L5-S1 facet joint osteoarthritis ≥ moderate</li>
                    <li>L3-4 EBQ > 3.6 AND sagittal imbalance present</li>
                    <li>Modic changes ≥ Type 2 AND significant spinal stenosis</li>
                </ul>
                <p>💡 L5-S1 segment risk is mainly related to facet joint degeneration</p>
            </div>
            """, unsafe_allow_html=True)
        else:  # Low risk
            st.markdown("""
            <div class="success-box">
                <h4>📋 Research-Based Typical Low-Risk Features:</h4>
                <ul>
                    <li>Normal L5-S1 facet joint function</li>
                    <li>Good bone density and no significant degeneration</li>
                    <li>Normal spinal alignment without imbalance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


def create_patient_input_form():
    """Create patient information input form"""
    st.markdown('<div class="section-header">🩺 Patient Information Input</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>DISCLAIMER</strong>: This prediction tool is for academic research only and cannot replace professional medical diagnosis.
        Prediction results are for reference only. All medical decisions should be made by qualified medical professionals.
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📋 Basic Information")
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            age = st.number_input("Age", min_value=50, max_value=80, value=60)
            bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=25.0, step=0.1)

            st.subheader("🏥 Comorbidities")
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            smoking = st.selectbox("Smoking History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            alcohol = st.selectbox("Alcohol Abuse", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col2:
            st.subheader("🦴 L3-4 Segment")
            l34_pfirrmann = st.selectbox("L3-4 Pfirrmann Grade", [1, 2, 3, 4, 5])
            l34_stenosis = st.selectbox("L3-4 Spinal Canal Stenosis", [0, 1, 2, 3],
                                        format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
            l34_foraminal = st.selectbox("L3-4 Foraminal Stenosis", [0, 1, 2, 3],
                                         format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
            l34_modic = st.selectbox("L3-4 Modic Changes", [0, 1, 2, 3])
            l34_facet = st.selectbox("L3-4 Facet Joint Osteoarthritis", [0, 1, 2, 3],
                                     format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
            l34_sagittal = st.selectbox("L3-4 Sagittal Imbalance", [0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
            l34_coronal = st.selectbox("L3-4 Coronal Imbalance", [0, 1],
                                       format_func=lambda x: "No" if x == 0 else "Yes")
            l34_ebq = st.number_input("L3-4 EBQ", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
            l34_lordosis = st.number_input("L3-4 Local Lordosis Angle", min_value=0.0, max_value=20.0, value=7.0,
                                           step=0.1)
            l34_disc_height = st.selectbox("L3-4 Preoperative Disc Height", [0, 1],
                                           format_func=lambda x: "Decreased" if x == 0 else "Normal")

        with col3:
            st.subheader("🦴 L5-S1 Segment")
            l5s1_pfirrmann = st.selectbox("L5-S1 Pfirrmann Grade", [1, 2, 3, 4, 5])
            l5s1_stenosis = st.selectbox("L5-S1 Spinal Canal Stenosis", [0, 1, 2, 3],
                                         format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
            l5s1_foraminal = st.selectbox("L5-S1 Foraminal Stenosis", [0, 1, 2, 3],
                                          format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
            l5s1_modic = st.selectbox("L5-S1 Modic Changes", [0, 1, 2, 3])
            l5s1_facet = st.selectbox("L5-S1 Facet Joint Osteoarthritis", [0, 1, 2, 3],
                                      format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
            l5s1_sagittal = st.selectbox("L5-S1 Sagittal Imbalance", [0, 1],
                                         format_func=lambda x: "No" if x == 0 else "Yes")
            l5s1_coronal = st.selectbox("L5-S1 Coronal Imbalance", [0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
            l5s1_ebq = st.number_input("L5-S1 EBQ", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
            l5s1_lordosis = st.number_input("L5-S1 Local Lordosis Angle", min_value=0.0, max_value=25.0, value=14.0,
                                            step=0.1)
            l5s1_disc_height = st.selectbox("L5-S1 Preoperative Disc Height", [0, 1],
                                            format_func=lambda x: "Decreased" if x == 0 else "Normal")

            st.subheader("⚕️ Surgical Parameters")
            cage_height = st.selectbox("Cage Height", [10, 12, 14])
            operative_time = st.number_input("Operative Time (minutes)", min_value=40, max_value=300, value=150)
            blood_loss = st.number_input("Blood Loss (ml)", min_value=20, max_value=400, value=150)

            st.subheader("📊 Other Parameters")
            hu = st.number_input("Bone Density HU Value", min_value=80.0, max_value=200.0, value=130.0, step=1.0)
            l3s1_lordosis = st.number_input("L3-S1 Lordosis Angle", min_value=20.0, max_value=35.0, value=28.0,
                                            step=0.1)
            lumbar_lordosis = st.number_input("Lumbar Lordosis Angle", min_value=25.0, max_value=45.0, value=34.0,
                                              step=0.1)

        submitted = st.form_submit_button("🔍 Start Prediction", use_container_width=True)

        if submitted:
            patient_data = {
                'Gender': gender,
                'Hypertension': hypertension,
                'Diabetes': diabetes,
                'Smoking history': smoking,
                'Alcohol abuse': alcohol,
                'L3-4 pfirrmann grade': l34_pfirrmann,
                'L3-4 spinal canal stenosis': l34_stenosis,
                'L3-4 foraminal stenosis': l34_foraminal,
                'L3-4 modic change': l34_modic,
                'L3-4 osteoarthritis of facet joints': l34_facet,
                'L3-4 sagittal imbalance': l34_sagittal,
                'L3-4 coronal imbalance': l34_coronal,
                'L5-S1 pfirrmann grade': l5s1_pfirrmann,
                'L5-S1 spinal canal stenosis': l5s1_stenosis,
                'L5-S1 foraminal stenosis': l5s1_foraminal,
                'L5-S1 modic change': l5s1_modic,
                'L5-S1 osteoarthritis of facet joints': l5s1_facet,
                'L5-S1 sagittal imbalance': l5s1_sagittal,
                'L5-S1 coronal imbalance': l5s1_coronal,
                'Cage height': cage_height,
                'Age': age,
                'BMI': bmi,
                'HU': hu,
                'L3-4 EBQ': l34_ebq,
                'L3-4 local lordosis angle': l34_lordosis,
                'L5-S1 EBQ': l5s1_ebq,
                'L5-S1 local lordosis angle': l5s1_lordosis,
                'L3-S1 lordosis angle': l3s1_lordosis,
                'Lumbar lordosis angle': lumbar_lordosis,
                'L3-4 preoperative disc height': l34_disc_height,
                'L5-S1 preoperative disc height': l5s1_disc_height,
                'Operative time ': operative_time,
                'Blood loss': blood_loss
            }

            return patient_data

    return None


def predict_with_explanations(loader, patient_data):
    """Predict for new patient and generate explanations"""
    st.markdown('<div class="main-header">🎯 Prediction Results & Explanations</div>', unsafe_allow_html=True)

    # Display disclaimer again
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>IMPORTANT REMINDER</strong>: Prediction results are for academic research reference only and cannot replace professional medical judgment!
        All medical decisions should be made by qualified medical professionals.
    </div>
    """, unsafe_allow_html=True)

    try:
        segments = ["L34", "L5S1"]

        for segment in segments:
            segment_name = "L3-4" if segment == "L34" else "L5-S1"
            st.markdown(f'<div class="section-header">{segment_name} Segment Complete Analysis</div>',
                        unsafe_allow_html=True)

            # Get feature order and build sample
            feature_names = loader.data_info[segment]['feature_names']
            sample_data = []
            for feature in feature_names:
                if feature in patient_data:
                    sample_data.append(patient_data[feature])
                else:
                    sample_data.append(0)

            sample_df = pd.DataFrame([sample_data], columns=feature_names)
            sample_array = sample_df.values[0]

            # 1. APLR prediction
            aplr_model = loader.aplr_models[segment]
            prediction = int(aplr_model.predict(sample_df)[0])
            probability = aplr_model.predict_class_probabilities(sample_df)[0]

            # Display basic prediction results
            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Prediction Result</h4>
                        <p>This patient may develop ASD in the <strong>{}</strong> segment</p>
                    </div>
                    """.format(segment_name), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <h4>✅ Prediction Result</h4>
                        <p>This patient has low probability of developing ASD in the <strong>{}</strong> segment</p>
                    </div>
                    """.format(segment_name), unsafe_allow_html=True)

                st.write(f"**ASD Probability**: {probability[1]:.3f}")
                st.write(f"**Non-ASD Probability**: {probability[0]:.3f}")

            with col2:
                # Probability visualization
                fig = go.Figure(data=[
                    go.Bar(x=['Non-ASD', 'ASD'], y=[probability[0], probability[1]],
                           marker_color=['#2ecc71' if probability[0] > probability[1] else '#a8e6cf',
                                         '#e74c3c' if probability[1] > probability[0] else '#ffcccb'])
                ])
                fig.update_layout(
                    title=f"{segment_name} Segment ASD Probability",
                    yaxis_title="Probability",
                    showlegend=False,
                    height=300,
                    title_x=0.5,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
                fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
                st.plotly_chart(fig, use_container_width=True)

            # 2. Local feature contribution analysis
            st.markdown("### 📊 Local Feature Contribution Analysis")
            fig_local, local_df = generate_local_explanation_plots(loader, segment, sample_df, feature_names)

            if fig_local:
                st.plotly_chart(fig_local, use_container_width=True)

                with st.expander("📋 View Detailed Local Contribution Data"):
                    st.dataframe(local_df.sort_values('Abs_Contribution', ascending=False), use_container_width=True)
            else:
                st.warning("Unable to generate local feature contribution plot")

            # 3. Anchor explanations
            st.markdown("### 🎯 Anchor Explanations")
            try:
                anchor_explainer = loader.create_anchor_explainer(segment)
                if anchor_explainer:
                    with st.spinner("Generating anchor explanations..."):
                        anchor_explanation = anchor_explainer.explain(
                            sample_array,
                            threshold=0.85,
                            delta=0.15,
                            tau=0.2,
                            batch_size=50,
                            coverage_samples=200,
                            beam_size=1
                        )

                        if len(anchor_explanation.anchor) == 0:
                            st.markdown("""
                            <div class="info-box">
                                <h4>🔄 Stable Prediction</h4>
                                <p>The model's prediction for this sample is very stable, no specific conditions needed to determine the prediction result</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>📋 Key Decision Rules</h4>
                                <p><strong>Precision</strong>: {anchor_explanation.precision:.3f} | <strong>Coverage</strong>: {anchor_explanation.coverage:.3f}</p>
                                <p>When the patient simultaneously meets the following conditions, the model prediction result remains unchanged:</p>
                            </div>
                            """, unsafe_allow_html=True)

                            for i, rule in enumerate(anchor_explanation.anchor):
                                st.write(f"   {i + 1}. {rule}")

                            st.info(
                                f"💡 This rule applies to {anchor_explanation.coverage:.1%} of the patient population, "
                                f"with {anchor_explanation.precision:.1%} receiving the same prediction result")
                else:
                    st.warning("⚠️ Anchor explainer creation failed, showing research-based typical rules")
                    show_typical_anchor_rules(segment, prediction)

            except Exception as e:
                st.warning(
                    f"⚠️ Anchor explanation generation encountered technical issues, showing research-based typical rules")
                show_typical_anchor_rules(segment, prediction)

            # 4. Counterfactual analysis
            st.markdown("### 🔄 Counterfactual Analysis")

            # Attempt real counterfactual analysis
            counterfactual_result = attempt_counterfactual_analysis(loader, segment, sample_array, prediction,
                                                                    feature_names)

            if counterfactual_result['success']:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Counterfactual Analysis Successful</h4>
                    <p><strong>Found parameter adjustments that can change prediction result:</strong></p>
                </div>
                """, unsafe_allow_html=True)

                changes_df = pd.DataFrame(counterfactual_result['changes'])
                st.dataframe(changes_df, use_container_width=True)

                target_desc = "low risk" if counterfactual_result['target_class'] == 0 else "high risk"
                st.info(f"💡 Through the above parameter adjustments, the prediction result may change to {target_desc}")

            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Counterfactual Analysis Failed</h4>
                    <p><strong>Failure Reason</strong>: {counterfactual_result['reason']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Display current surgical parameter status
                st.write("**Current Surgical Parameters:**")
                interventional_features = ['Cage height', 'Operative time ', 'Blood loss']
                current_params = []

                for feature in interventional_features:
                    if feature in patient_data:
                        current_params.append({
                            'Parameter': feature,
                            'Current Value': patient_data[feature],
                            'Status': 'Normal Range' if is_parameter_in_normal_range(feature, patient_data[
                                feature]) else 'Needs Attention'
                        })

                if current_params:
                    params_df = pd.DataFrame(current_params)
                    st.dataframe(params_df, use_container_width=True)

                st.markdown("""
                <div class="info-box">
                    <h4>💡 Since no effective counterfactual explanation was found, recommendations include:</h4>
                    <ul>
                        <li>Strictly follow clinical guidelines for surgery</li>
                        <li>Focus on patient's non-modifiable risk factors</li>
                        <li>Strengthen postoperative follow-up and rehabilitation guidance</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

    except Exception as e:
        st.error(f"Prediction process error: {e}")


def main():
    """Main function"""
    # Load models
    loader = get_model_loader()
    success = loader.load_all_models()

    if not success:
        st.error("Model loading failed, please ensure save_models.py has been run to save models")
        return

    # Sidebar navigation
    st.sidebar.markdown('<div class="section-header">📋 Navigation Menu</div>', unsafe_allow_html=True)

    # Sidebar disclaimer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="warning-box">
        <h4>⚠️ DISCLAIMER</h4>
        <p>This system is for academic research only and cannot replace professional medical judgment. Prediction results are for reference only.</p>
    </div>
    """, unsafe_allow_html=True)

    menu_options = ["🏠 Home", "🔬 Research Results", "🩺 Smart Prediction"]
    choice = st.sidebar.selectbox("Select Function", menu_options)

    if choice == "🏠 Home":
        show_homepage()

    elif choice == "🔬 Research Results":
        show_research_results(loader)

        # Add model summary information
        st.markdown("---")
        region_for_summary = st.selectbox("Select Model Summary", ["L34", "L5S1"],
                                          format_func=lambda x: "L3-4" if x == "L34" else "L5-S1", key="summary_region")
        show_aplr_model_summary(loader, region_for_summary)

    elif choice == "🩺 Smart Prediction":
        st.markdown('<div class="main-header">🩺 Intelligent ASD Risk Prediction & Explanation</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Input patient information to get risk assessment, feature contribution analysis, anchor explanations, and counterfactual analysis")

        patient_data = create_patient_input_form()

        if patient_data:
            predict_with_explanations(loader, patient_data)


if __name__ == "__main__":
    main()
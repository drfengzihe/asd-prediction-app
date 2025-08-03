import pandas as pd
import numpy as np
import pickle
import joblib
from interpret.glassbox import APLRClassifier
from alibi.explainers import AnchorTabular
import streamlit as st
import os


class ASDPredictor:
    def __init__(self):
        self.l34_model = None
        self.l5s1_model = None
        self.feature_names = None
        self.data_path = "data/L3-4近端和L5-S1远端合并版.xlsx"

    @st.cache_data
    def load_data(_self):
        """加载数据"""
        try:
            l34_data = pd.read_excel(_self.data_path, sheet_name="L34 Result")
            l5s1_data = pd.read_excel(_self.data_path, sheet_name="L5S1 Result")
            return l34_data, l5s1_data
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            return None, None

    def train_models(self):
        """训练模型"""
        l34_data, l5s1_data = self.load_data()

        if l34_data is None or l5s1_data is None:
            return False

        # 训练L3-4模型
        X_l34 = l34_data.drop('Result', axis=1)
        y_l34 = l34_data['Result']
        self.feature_names = X_l34.columns.tolist()

        self.l34_model = APLRClassifier(random_state=42)
        self.l34_model.fit(X_l34, y_l34, X_names=self.feature_names)

        # 训练L5-S1模型
        X_l5s1 = l5s1_data.drop('Result', axis=1)
        y_l5s1 = l5s1_data['Result']

        self.l5s1_model = APLRClassifier(random_state=42)
        self.l5s1_model.fit(X_l5s1, y_l5s1, X_names=self.feature_names)

        return True

    def predict_new_sample(self, sample_data, segment="L3-4"):
        """预测新样本"""
        if segment == "L3-4" and self.l34_model:
            model = self.l34_model
        elif segment == "L5-S1" and self.l5s1_model:
            model = self.l5s1_model
        else:
            return None, None

        # 确保特征顺序正确
        sample_df = pd.DataFrame([sample_data])
        sample_df = sample_df.reindex(columns=self.feature_names, fill_value=0)

        prediction = int(model.predict(sample_df)[0])
        probability = model.predict_class_probabilities(sample_df)[0]

        return prediction, probability
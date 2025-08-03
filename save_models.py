#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import joblib
from interpret.glassbox import APLRClassifier
from interpret import show, set_visualize_provider
from interpret.provider import InlineProvider
from alibi.explainers import AnchorTabular, CounterfactualProto
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings('ignore')

# TensorFlow配置
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

tf.get_logger().setLevel(40)
tf.compat.v1.disable_v2_behavior()

# 配置
set_visualize_provider(InlineProvider())
SEED = 42
np.random.seed(SEED)

# 数据路径配置
DATA_PATH = "data/L3-4近端和L5-S1远端合并版.xlsx"
L34_SHEET = "L34 Result"
L5S1_SHEET = "L5S1 Result"


class APLRPredictor:
    """APLR模型预测器类，可以被pickle序列化"""

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
    """随机森林预测器类，可以被pickle序列化"""

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


class EnhancedModelSaver:
    """保存训练好的模型和所有解释组件"""

    def __init__(self):
        self.interventional_features = ['Cage height', 'Operative time ', 'Blood loss']
        self.feature_constraints = {
            'Cage height': (8.0, 16.0),
            'Operative time ': (0.0, 600.0),
            'Blood loss': (0.0, 1000.0)
        }

    def identify_categorical_features(self, X):
        """识别分类特征"""
        categorical_features = {}
        categorical_indices = []

        for i, col in enumerate(X.columns):
            if X[col].dtype in ['int64', 'int32', 'object'] or X[col].nunique() <= 10:
                unique_vals = sorted(X[col].unique())
                categorical_features[i] = [str(val) for val in unique_vals]
                categorical_indices.append(i)

        return categorical_features, categorical_indices

    def build_autoencoder(self, input_dim, latent_dim=16):
        """构建自编码器"""
        input_layer = tf.keras.layers.Input(shape=(input_dim,))

        # 编码器
        encoded = tf.keras.layers.Dense(32, activation='relu')(input_layer)
        encoded = tf.keras.layers.Dense(latent_dim, activation='relu')(encoded)

        # 解码器
        decoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)

        # 构建模型
        autoencoder = tf.keras.models.Model(input_layer, decoded)
        encoder = tf.keras.models.Model(input_layer, encoded)

        # 编译模型
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        autoencoder.compile(optimizer=optimizer, loss='mse')

        return autoencoder, encoder

    def train_and_save_all(self):
        """训练并保存所有模型和解释组件"""
        print("开始训练并保存所有模型和解释组件...")

        # 确保models文件夹存在
        os.makedirs("models", exist_ok=True)

        for sheet_name, region in [(L34_SHEET, "L34"), (L5S1_SHEET, "L5S1")]:
            print(f"\n处理{region}数据集...")

            # 加载数据
            data = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
            X = data.drop('Result', axis=1)
            y = data['Result'].astype(int)
            feature_names = X.columns.tolist()

            print(f"数据形状: {data.shape}")
            print(f"ASD分布: {y.value_counts().to_dict()}")

            # 1. 训练APLR模型
            print("训练APLR模型...")
            aplr_model = APLRClassifier(random_state=SEED)
            aplr_model.fit(X, y, X_names=feature_names)

            train_pred = np.array([int(p) for p in aplr_model.predict(X)])
            aplr_accuracy = np.mean(train_pred == y)
            print(f"APLR准确率: {aplr_accuracy:.4f}")

            # 2. 训练随机森林模型（用于反事实分析）
            print("训练随机森林模型...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
            rf_model.fit(X_train_scaled, y_train)

            rf_pred = rf_model.predict(X_test_scaled)
            rf_accuracy = np.mean(rf_pred == y_test)
            print(f"随机森林准确率: {rf_accuracy:.4f}")

            # 3. 训练自编码器
            print("训练自编码器...")
            autoencoder, encoder = self.build_autoencoder(X_train_scaled.shape[1], latent_dim=16)
            autoencoder.fit(
                X_train_scaled, X_train_scaled,
                epochs=50, batch_size=24,  # 减少epochs避免过长训练时间
                validation_split=0.2, verbose=0
            )

            # 4. 设置锚点解释器
            print("设置锚点解释器...")
            categorical_features, _ = self.identify_categorical_features(X)

            # 创建可序列化的预测器
            aplr_predictor = APLRPredictor(aplr_model, feature_names)

            anchor_explainer = AnchorTabular(
                predictor=aplr_predictor,
                feature_names=feature_names,
                categorical_names=categorical_features if categorical_features else None
            )

            anchor_explainer.fit(X.values, disc_perc=[25, 50, 75])

            # 5. 设置反事实解释器
            print("设置反事实解释器...")
            rf_predictor = RFPredictor(rf_model, scaler)

            cf_explainer = CounterfactualProto(
                rf_predictor,
                shape=(1, X_train.shape[1]),
                beta=0.1,
                gamma=100.0,
                theta=100.0,
                ae_model=autoencoder,
                enc_model=encoder,
                max_iterations=200,  # 减少迭代次数
                feature_range=(np.min(X_train.values, axis=0), np.max(X_train.values, axis=0)),
                c_init=1.0,
                c_steps=3,
                learning_rate_init=0.01,
                clip=(-1000.0, 1000.0)
            )

            print("拟合反事实解释器...")
            cf_explainer.fit(X_train_scaled)

            # 6. 生成全局和局部解释
            print("生成解释结果...")
            global_exp = aplr_model.explain_global(name=f"{region} 全局解释")

            # 选择代表性样本生成局部解释
            high_risk_indices = np.where(y == 1)[0][:3]
            low_risk_indices = np.where(y == 0)[0][:3]
            selected_indices = np.concatenate([high_risk_indices, low_risk_indices])

            local_explanations = []
            for idx in selected_indices:
                sample_X = X.iloc[[idx]]
                sample_y = y.iloc[idx]

                local_exp = aplr_model.explain_local(
                    sample_X.values,
                    np.array([sample_y]),
                    name=f"{region} 样本{idx} 局部解释"
                )

                local_explanations.append({
                    'index': idx,
                    'explanation': local_exp,
                    'true_label': sample_y,
                    'features': sample_X.iloc[0].to_dict()
                })

            # 7. 保存所有组件
            print("保存所有组件...")

            # 保存基础模型
            joblib.dump(aplr_model, f"models/{region}_aplr_model.joblib")
            joblib.dump(rf_model, f"models/{region}_rf_model.joblib")
            joblib.dump(scaler, f"models/{region}_scaler.joblib")

            # 保存自编码器（使用TensorFlow的保存方式）
            autoencoder.save_weights(f"models/{region}_autoencoder_weights.h5")
            encoder.save_weights(f"models/{region}_encoder_weights.h5")

            # 保存模型架构信息
            autoencoder_config = {
                'input_dim': X_train_scaled.shape[1],
                'latent_dim': 16
            }
            with open(f"models/{region}_autoencoder_config.pkl", 'wb') as f:
                pickle.dump(autoencoder_config, f)

            # 保存解释器组件（分别保存，避免pickle问题）
            try:
                # 保存锚点解释器的训练数据和配置
                anchor_data = {
                    'categorical_features': categorical_features,
                    'feature_names': feature_names,
                    'training_data': X.values,
                    'disc_perc': [25, 50, 75]
                }
                with open(f"models/{region}_anchor_data.pkl", 'wb') as f:
                    pickle.dump(anchor_data, f)

                print(f"锚点解释器数据已保存")
            except Exception as e:
                print(f"锚点解释器保存失败: {e}")

            # 保存解释结果
            with open(f"models/{region}_global_explanation.pkl", 'wb') as f:
                pickle.dump(global_exp, f)

            with open(f"models/{region}_local_explanations.pkl", 'wb') as f:
                pickle.dump(local_explanations, f)

            # 保存数据信息
            data_info = {
                'feature_names': feature_names,
                'data_shape': data.shape,
                'asd_distribution': y.value_counts().to_dict(),
                'aplr_accuracy': aplr_accuracy,
                'rf_accuracy': rf_accuracy,
                'interventional_features': self.interventional_features,
                'feature_constraints': self.feature_constraints
            }

            with open(f"models/{region}_data_info.pkl", 'wb') as f:
                pickle.dump(data_info, f)

            # 保存训练数据
            training_data = {
                'X': X, 'y': y,
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled
            }
            with open(f"models/{region}_training_data.pkl", 'wb') as f:
                pickle.dump(training_data, f)

            print(f"{region} 所有组件保存完成")

        print("\n🎉 所有模型和解释组件保存完成！")
        print("保存的文件：")
        for file in sorted(os.listdir("models")):
            if file.endswith(('.pkl', '.joblib', '.h5')):
                print(f"  - models/{file}")


if __name__ == "__main__":
    saver = EnhancedModelSaver()
    saver.train_and_save_all()
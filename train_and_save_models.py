# train_and_save_models.py
import pandas as pd
import numpy as np
import pickle
import os
from interpret.glassbox import APLRClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.sample_data = {}
        self.categorical_features = {}

    def load_and_train(self, data_path):
        """训练模型并保存所有必要组件"""

        # 加载数据
        l34_data = pd.read_excel(data_path, sheet_name="L34 Result")
        l5s1_data = pd.read_excel(data_path, sheet_name="L5S1 Result")

        datasets = {
            'l34': l34_data,
            'l5s1': l5s1_data
        }

        for name, data in datasets.items():
            print(f"训练 {name.upper()} 模型...")

            # 预处理数据
            X = data.drop('Result', axis=1)
            y = data['Result'].astype(int)

            # 保存特征名称
            self.feature_names[name] = X.columns.tolist()

            # 识别并保存分类特征信息
            categorical_features = {}
            for i, col in enumerate(X.columns):
                if X[col].nunique() <= 10:
                    categorical_features[i] = [str(val) for val in sorted(X[col].unique())]
            self.categorical_features[name] = categorical_features

            # 保存统计信息而不是原始数据
            self.sample_data[name] = {
                'feature_stats': {
                    'mean': X.mean().to_dict(),
                    'std': X.std().to_dict(),
                    'min': X.min().to_dict(),
                    'max': X.max().to_dict(),
                    'median': X.median().to_dict()
                },
                'target_distribution': y.value_counts().to_dict(),
                'sample_size': len(X),
                'X_sample': X.values,  # 保存数据用于锚点解释器训练
                'y_sample': y.values
            }

            # 训练APLR模型
            aplr_model = APLRClassifier(random_state=42)
            aplr_model.fit(X, y, X_names=self.feature_names[name])
            self.models[name] = aplr_model

            # 设置标准化器
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[name] = scaler

            print(f"{name.upper()} 模型训练完成")

    def save_models(self, save_dir='models'):
        """保存所有模型和组件"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存APLR模型
        with open(f'{save_dir}/aplr_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)

        # 保存标准化器
        with open(f'{save_dir}/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)

        # 保存特征名称
        with open(f'{save_dir}/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)

        # 保存分类特征信息
        with open(f'{save_dir}/categorical_features.pkl', 'wb') as f:
            pickle.dump(self.categorical_features, f)

        # 保存样本数据
        with open(f'{save_dir}/sample_data.pkl', 'wb') as f:
            pickle.dump(self.sample_data, f)

        print(f"所有模型已保存到 {save_dir} 目录")


# 使用示例
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.load_and_train(r"G:\OneDrive\科研\asd\L3-4近端和L5-S1远端合并版.xlsx")  # 使用你的实际路径
    trainer.save_models()
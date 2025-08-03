#!/usr/bin/env python
# coding: utf-8

# In[12]:


#!/usr/bin/env python
# coding: utf-8

"""
ASD预测模型的反事实推理分析
专注于反事实解释功能
"""

import os
import warnings
warnings.filterwarnings('ignore')

# 设置TensorFlow环境
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
tf.get_logger().setLevel(40)
tf.compat.v1.disable_v2_behavior()

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 导入 Alibi 反事实解释器
from alibi.explainers import CounterfactualProto

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CounterfactualAnalyzer:
    """ASD预测模型反事实分析器"""
    
    def __init__(self, data_path="G:/OneDrive/科研/asd/L3-4近端和L5-S1远端合并版.xlsx"):
        """初始化分析器"""
        self.data_path = data_path
        self.interventional_features = ['Cage height', 'Operative time ', 'Blood loss']
        self.feature_constraints = {
            'Cage height': (8.0, 16.0),
            'Operative time ': (0.0, 600.0),
            'Blood loss': (0.0, 1000.0)
        }
        
        # 初始化数据容器
        self.processed_data = {}
        self.models = {}
        self.scalers = {}
        self.autoencoders = {}
        self.cf_explainers = {}
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("正在加载和预处理数据...")
        
        # 加载数据
        l34_data = pd.read_excel(self.data_path, sheet_name="L34 Result", header=0)
        l5s1_data = pd.read_excel(self.data_path, sheet_name="L5S1 Result", header=0)
        
        print(f"L34数据集形状: {l34_data.shape}")
        print(f"L5S1数据集形状: {l5s1_data.shape}")
        
        # 预处理数据
        for region_name, data in [("l34", l34_data), ("l5s1", l5s1_data)]:
            # 提取特征和目标变量
            X = data.drop('Result', axis=1)
            y = data['Result']
            feature_names = X.columns.tolist()
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 保存处理后的数据
            self.processed_data[region_name] = {
                'X_train': X_train.values,
                'X_test': X_test.values,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'y_train': y_train.values,
                'y_test': y_test.values,
                'feature_names': feature_names
            }
            self.scalers[region_name] = scaler
        
        print("数据预处理完成")
        return self
    
    def train_models(self):
        """训练预测模型"""
        print("正在训练预测模型...")
        
        for region_name in ["l34", "l5s1"]:
            data = self.processed_data[region_name]
            
            # 训练随机森林模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(data['X_train_scaled'], data['y_train'])
            
            # 评估模型
            y_pred = model.predict(data['X_test_scaled'])
            accuracy = accuracy_score(data['y_test'], y_pred)
            print(f"{region_name.upper()}模型准确率: {accuracy:.4f}")
            
            self.models[region_name] = model
        
        print("模型训练完成")
        return self
    
    def build_autoencoder(self, input_dim, latent_dim=16):
        """构建自编码器模型"""
        # 输入层
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
    
    def train_autoencoders(self):
        """训练自编码器"""
        print("正在训练自编码器...")
        
        for region_name in ["l34", "l5s1"]:
            data = self.processed_data[region_name]
            
            # 构建自编码器
            autoencoder, encoder = self.build_autoencoder(
                data['X_train_scaled'].shape[1], latent_dim=16
            )
            
            # 训练自编码器
            print(f"训练{region_name.upper()}自编码器...")
            history = autoencoder.fit(
                data['X_train_scaled'], 
                data['X_train_scaled'],
                epochs=200, 
                batch_size=24, 
                validation_split=0.2,
                verbose=0
            )
            
            self.autoencoders[region_name] = {
                'autoencoder': autoencoder,
                'encoder': encoder,
                'history': history
            }
        
        print("自编码器训练完成")
        return self
    
    def create_cf_predictor(self, region_name):
        """创建反事实预测函数"""
        model = self.models[region_name]
        scaler = self.scalers[region_name]
        
        def predictor(x):
            """反事实预测函数，返回类别概率"""
            if isinstance(x, np.ndarray):
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                # 获取预测标签
                pred_labels = model.predict(scaler.transform(x))
                # 将标签转换为概率形式
                pred_proba = np.zeros((len(pred_labels), 2))
                for i, label in enumerate(pred_labels):
                    pred_proba[i, int(label)] = 1.0
                return pred_proba
            else:
                raise ValueError("输入必须是NumPy数组")
        
        return predictor
    
    def build_counterfactual_explainers(self):
        """构建反事实解释器"""
        print("正在构建反事实解释器...")
        
        for region_name in ["l34", "l5s1"]:
            data = self.processed_data[region_name]
            autoencoder_info = self.autoencoders[region_name]
            
            # 创建预测函数
            cf_predictor = self.create_cf_predictor(region_name)
            
            # 创建反事实解释器
            cf_explainer = CounterfactualProto(
                cf_predictor,
                shape=(1, data['X_train'].shape[1]),
                beta=0.1,                    # L1正则化系数
                gamma=100.0,                 # 自编码器损失系数
                theta=100.0,                 # 原型损失系数
                ae_model=autoencoder_info['autoencoder'],
                enc_model=autoencoder_info['encoder'],
                max_iterations=500,          # 最大迭代次数
                feature_range=(              # 特征值范围
                    np.min(data['X_train'], axis=0),
                    np.max(data['X_train'], axis=0)
                ),
                c_init=1.0,                  # 初始化c参数
                c_steps=5,                   # c参数更新步数
                learning_rate_init=0.01,     # 初始学习率
                clip=(-1000.0, 1000.0)       # 梯度裁剪范围
            )
            
            # 拟合反事实解释器
            print(f"拟合{region_name.upper()}反事实解释器...")
            cf_explainer.fit(data['X_train_scaled'])
            
            self.cf_explainers[region_name] = cf_explainer
        
        print("反事实解释器构建完成")
        return self
    
    def select_samples_by_prediction(self, region_name, prediction_class, n_samples=5):
        """根据预测结果选择样本"""
        data = self.processed_data[region_name]
        model = self.models[region_name]
        
        # 获取测试集预测结果
        y_pred = model.predict(data['X_test_scaled'])
        
        # 选择指定预测类别的样本
        mask = (y_pred == prediction_class)
        filtered_X = data['X_test'][mask]
        
        if len(filtered_X) == 0:
            print(f"没有找到{region_name.upper()}区域预测为{prediction_class}的样本")
            return []
        
        if len(filtered_X) <= n_samples:
            print(f"{region_name.upper()}区域只找到{len(filtered_X)}个预测为{prediction_class}的样本")
            return filtered_X
        
        # 随机选择n_samples个样本
        indices = np.random.choice(len(filtered_X), n_samples, replace=False)
        return filtered_X[indices]
    
    def generate_counterfactual_for_sample(self, region_name, sample, target_class, sample_id):
        """为单个样本生成反事实"""
        cf_explainer = self.cf_explainers[region_name]
        scaler = self.scalers[region_name]
        feature_names = self.processed_data[region_name]['feature_names']
        
        # 确保样本是2D数组
        if len(sample.shape) == 1:
            sample_2d = sample.reshape(1, -1)
        else:
            sample_2d = sample
        
        # 标准化样本
        sample_scaled = scaler.transform(sample_2d)
        
        try:
            # 生成反事实解释
            explanation = cf_explainer.explain(sample_scaled, target_class=[target_class])
            
            if explanation.cf is not None and 'X' in explanation.cf:
                # 提取反事实实例
                cf_instance = explanation.cf['X']
                
                # 反标准化回原始尺度
                cf_instance = scaler.inverse_transform(cf_instance)
                
                # 检查特征约束
                violated_constraints = False
                for i, feature in enumerate(feature_names):
                    if feature in self.feature_constraints:
                        min_val, max_val = self.feature_constraints[feature]
                        if cf_instance[0, i] < min_val or cf_instance[0, i] > max_val:
                            violated_constraints = True
                            print(f"反事实违反了{feature}的约束 ({min_val:.1f}, {max_val:.1f})")
                            break
                
                if violated_constraints:
                    print(f"样本{sample_id}在{region_name.upper()}的反事实违反约束，被拒绝")
                    return explanation, {}
                
                # 计算可干预特征的变化
                changes = {}
                for i, feature in enumerate(feature_names):
                    if feature in self.interventional_features:
                        original_value = sample_2d[0, i]
                        cf_value = cf_instance[0, i]
                        
                        if abs(original_value - cf_value) > 1e-6:
                            changes[feature] = {
                                'original': original_value,
                                'counterfactual': cf_value,
                                'change': cf_value - original_value
                            }
                
                if changes:
                    target_desc = "高风险" if target_class == 1 else "低风险"
                    print(f"\n样本 {sample_id} 在 {region_name.upper()} (目标: {target_desc}):")
                    print(f"原始预测: {explanation.orig_class}")
                    print(f"反事实预测: {explanation.cf['class']}")
                    print("需要的变化:")
                    for feature, values in changes.items():
                        print(f"  {feature}: {values['original']:.2f} → {values['counterfactual']:.2f} (Δ: {values['change']:.2f})")
                    
                    return explanation, changes
            
            print(f"未找到样本{sample_id}在{region_name.upper()}的有效反事实")
            return explanation, {}
        
        except Exception as e:
            print(f"为样本{sample_id}在{region_name.upper()}生成反事实时出错: {str(e)}")
            return None, {}
    
    def analyze_counterfactual_patterns(self, region_name, results, direction):
        """分析反事实解释的特征变化模式"""
        if not results:
            print(f"没有{region_name.upper()} {direction}的反事实结果可分析")
            return
        
        # 收集所有特征变化
        feature_changes = {feature: [] for feature in self.interventional_features}
        
        for exp, changes in results:
            for feature, change_info in changes.items():
                if feature in self.interventional_features:
                    feature_changes[feature].append(change_info['change'])
        
        # 分析每个特征的变化模式
        print(f"\n{region_name.upper()} {direction} 反事实解释的特征变化模式:")
        for feature, changes in feature_changes.items():
            if changes:
                avg_change = np.mean(changes)
                std_change = np.std(changes)
                direction_text = "增加" if avg_change > 0 else "减少"
                
                print(f"  {feature}:")
                print(f"    变化方向: {direction_text}")
                print(f"    平均变化: {avg_change:.4f}")
                print(f"    标准差: {std_change:.4f}")
                print(f"    变化样本数: {len(changes)}")
            else:
                print(f"  {feature}: 无变化")
    
    def visualize_counterfactual_changes(self, region_name, results, direction):
        """可视化反事实解释的特征变化"""
        if not results:
            print(f"没有{region_name.upper()} {direction}的反事实结果可视化")
            return
        
        # 收集特征变化数据
        feature_changes = {feature: [] for feature in self.interventional_features}
        
        for exp, changes in results:
            for feature in self.interventional_features:
                if feature in changes:
                    feature_changes[feature].append(changes[feature]['change'])
                else:
                    feature_changes[feature].append(0)
        
        # 计算平均变化
        avg_changes = {feature: np.mean(changes) for feature, changes in feature_changes.items() if changes}
        
        if not avg_changes:
            print(f"没有{region_name.upper()} {direction}的特征变化数据")
            return
        
        # 创建柱状图
        plt.figure(figsize=(10, 6))
        features = list(avg_changes.keys())
        values = list(avg_changes.values())
        
        colors = ['red' if v < 0 else 'green' for v in values]
        
        plt.bar(features, values, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'{region_name.upper()} {direction} 反事实解释的平均特征变化')
        plt.xlabel('特征')
        plt.ylabel('平均变化')
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(values):
            plt.text(i, v + (0.01 * max(abs(min(values)), abs(max(values)))), 
                    f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        plt.show()
    
    def run_counterfactual_analysis(self, n_samples=5):
        """运行完整的反事实分析"""
        print("开始反事实分析...")
        
        results = {}
        
        for region_name in ["l34", "l5s1"]:
            results[region_name] = {'low_to_high': [], 'high_to_low': []}
            
            # 选择低风险样本，目标转换为高风险
            low_risk_samples = self.select_samples_by_prediction(region_name, 0, n_samples)
            print(f"\n为{region_name.upper()}低风险样本生成反事实解释（目标：高风险）...")
            for i, sample in enumerate(low_risk_samples):
                exp, changes = self.generate_counterfactual_for_sample(
                    region_name, sample, 1, i+1
                )
                if changes:
                    results[region_name]['low_to_high'].append((exp, changes))
            
            # 选择高风险样本，目标转换为低风险
            high_risk_samples = self.select_samples_by_prediction(region_name, 1, n_samples)
            print(f"\n为{region_name.upper()}高风险样本生成反事实解释（目标：低风险）...")
            for i, sample in enumerate(high_risk_samples):
                exp, changes = self.generate_counterfactual_for_sample(
                    region_name, sample, 0, i+1
                )
                if changes:
                    results[region_name]['high_to_low'].append((exp, changes))
        
        return results
    
    def visualize_autoencoder_training(self):
        """可视化自编码器训练过程"""
        plt.figure(figsize=(12, 5))
        
        for i, region_name in enumerate(["l34", "l5s1"]):
            plt.subplot(1, 2, i+1)
            history = self.autoencoders[region_name]['history']
            plt.plot(history.history['loss'], label='训练损失')
            plt.plot(history.history['val_loss'], label='验证损失')
            plt.title(f'{region_name.upper()} 自编码器训练')
            plt.xlabel('Epochs')
            plt.ylabel('损失')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = CounterfactualAnalyzer()
    
    # 运行完整流程
    analyzer.load_and_preprocess_data()
    analyzer.train_models()
    analyzer.train_autoencoders()
    analyzer.build_counterfactual_explainers()
    
    # 可视化自编码器训练过程
    analyzer.visualize_autoencoder_training()
    
    # 运行反事实分析
    cf_results = analyzer.run_counterfactual_analysis(n_samples=5)
    
    # 分析和可视化结果
    for region_name in ["l34", "l5s1"]:
        for direction in ["low_to_high", "high_to_low"]:
            direction_text = "低→高" if direction == "low_to_high" else "高→低"
            
            # 分析特征变化模式
            analyzer.analyze_counterfactual_patterns(
                region_name, cf_results[region_name][direction], direction_text
            )
            
            # 可视化特征变化
            analyzer.visualize_counterfactual_changes(
                region_name, cf_results[region_name][direction], direction_text
            )
    
    # 打印结果统计
    print("\n反事实解释结果统计:")
    for region_name in ["l34", "l5s1"]:
        low_to_high_count = len(cf_results[region_name]['low_to_high'])
        high_to_low_count = len(cf_results[region_name]['high_to_low'])
        print(f"{region_name.upper()}低→高: {low_to_high_count}/5")
        print(f"{region_name.upper()}高→低: {high_to_low_count}/5")


# In[ ]:





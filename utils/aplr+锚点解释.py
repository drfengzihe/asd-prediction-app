#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
简化版 APLR ASD预测系统 + Anchors解释
功能：数据加载、模型训练、APLR解释、Anchors解释
"""

import pandas as pd
import numpy as np
from interpret.glassbox import APLRClassifier
from interpret import show, set_visualize_provider
from interpret.provider import InlineProvider
from alibi.explainers import AnchorTabular
from alibi.utils import gen_category_map

# 配置
set_visualize_provider(InlineProvider())
SEED = 42
np.random.seed(SEED)

# 数据路径配置
DATA_PATH = r"G:\OneDrive\科研\asd\L3-4近端和L5-S1远端合并版.xlsx"
L34_SHEET = "L34 Result"
L5S1_SHEET = "L5S1 Result"

class ASDPredictorWithAnchors:
    """带锚点解释的ASD预测器"""
    
    def __init__(self, random_state=SEED):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.data = None
        self.anchor_explainer = None
        self.X_train = None
        self.y_train = None
        
    def load_and_preprocess_data(self, file_path, sheet_name):
        """加载和预处理数据"""
        print(f"正在加载数据: {sheet_name}")
        
        # 加载数据
        self.data = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"数据形状: {self.data.shape}")
        print(f"列名: {list(self.data.columns)}")
        print(f"目标变量分布:\n{self.data['Result'].value_counts()}")
        
        # 分离特征和标签
        X = self.data.drop('Result', axis=1)
        y = self.data['Result'].astype(int)
        self.feature_names = X.columns.tolist()
        
        print(f"特征数量: {len(self.feature_names)}")
        print(f"样本数量: {len(X)}")
        
        # 保存数据用于锚点解释
        self.X_train = X
        self.y_train = y
        
        return X, y
    
    def identify_categorical_features(self, X):
        """识别分类特征"""
        categorical_features = {}
        categorical_indices = []
        
        for i, col in enumerate(X.columns):
            # 检查是否为整数类型且唯一值较少
            if X[col].dtype in ['int64', 'int32', 'object'] or X[col].nunique() <= 10:
                unique_vals = sorted(X[col].unique())
                categorical_features[i] = [str(val) for val in unique_vals]
                categorical_indices.append(i)
                print(f"识别分类特征 {col} (索引 {i}): {unique_vals}")
        
        return categorical_features, categorical_indices
    
    def train_model(self, X, y):
        """训练APLR模型"""
        print("\n开始训练APLR模型...")
        
        self.model = APLRClassifier(random_state=self.random_state)
        self.model.fit(X, y, X_names=self.feature_names)
        
        print("模型训练完成")
        
        # 计算训练集准确率
        train_pred = np.array([int(p) for p in self.model.predict(X)])
        train_accuracy = np.mean(train_pred == y)
        print(f"训练集准确率: {train_accuracy:.4f}")
        
        return self.model
    
    def setup_anchor_explainer(self, X):
        """设置锚点解释器"""
        print("\n设置锚点解释器...")
        
        # 识别分类特征
        categorical_features, _ = self.identify_categorical_features(X)
        
        # 创建预测函数
        def predict_fn(x):
            if isinstance(x, np.ndarray):
                x_df = pd.DataFrame(x, columns=self.feature_names)
            else:
                x_df = x
            predictions = self.model.predict(x_df)
            return np.array([int(p) for p in predictions])
        
        # 初始化锚点解释器
        self.anchor_explainer = AnchorTabular(
            predictor=predict_fn,
            feature_names=self.feature_names,
            categorical_names=categorical_features if categorical_features else None
        )
        
        # 拟合解释器
        print("拟合锚点解释器...")
        # 将数据转换为numpy数组
        X_array = X.values
        self.anchor_explainer.fit(X_array, disc_perc=[25, 50, 75])
        
        print("锚点解释器设置完成")
        
        return self.anchor_explainer
    
    def explain_global(self, dataset_name):
        """生成全局解释"""
        if self.model is None:
            print("请先训练模型")
            return None
        
        print(f"\n生成 {dataset_name} 模型的全局解释...")
        global_explanation = self.model.explain_global(name=f"{dataset_name} 全局解释")
        show(global_explanation)
        
        return global_explanation
    
    def explain_sample_with_anchors(self, X, y, sample_idx, dataset_name):
        """
        对单个样本进行APLR局部解释和锚点解释
        """
        print(f"\n{'='*60}")
        print(f"样本 {sample_idx + 1} (行索引: {sample_idx}) 的详细解释")
        print(f"{'='*60}")
        
        # 获取样本数据
        sample_X = X.iloc[[sample_idx]]
        sample_y = y.iloc[sample_idx]
        
        # APLR预测
        aplr_pred_raw = self.model.predict(sample_X)
        aplr_pred = int(aplr_pred_raw[0])
        aplr_prob = self.model.predict_class_probabilities(sample_X)[0]
        
        # 显示基本信息
        print(f"真实标签: {'ASD' if sample_y == 1 else '无ASD'}")
        print(f"APLR预测: {'ASD' if aplr_pred == 1 else '无ASD'}")
        print(f"ASD概率: {aplr_prob[1]:.3f}")
        print(f"预测{'正确' if aplr_pred == sample_y else '错误'}")
        
        # 显示关键特征值
        print(f"\n关键特征值:")
        sample_features = sample_X.iloc[0]
        key_features = ['Gender', 'Age', 'BMI', 'Hypertension', 'Diabetes']
        for feat in key_features:
            if feat in sample_features.index:
                print(f"  {feat}: {sample_features[feat]}")
        
        # 1. APLR局部解释
        print(f"\n--- APLR局部解释 ---")
        aplr_local_explanation = self.model.explain_local(
            sample_X.values, 
            np.array([sample_y]),
            name=f"{dataset_name} 样本{sample_idx+1} APLR局部解释"
        )
        show(aplr_local_explanation)
        
        # 2. 锚点解释
        print(f"\n--- 锚点解释 ---")
        if self.anchor_explainer is None:
            print("锚点解释器未初始化，跳过锚点解释")
            return {
                'sample_idx': sample_idx,
                'aplr_prediction': aplr_pred,
                'aplr_probability': aplr_prob,
                'true_label': sample_y,
                'aplr_explanation': aplr_local_explanation,
                'anchor_explanation': None
            }
        
        try:
            # 获取锚点解释
            sample_array = sample_X.values[0]
            anchor_explanation = self.anchor_explainer.explain(
                sample_array,
                threshold=0.90,  # 90%精度阈值
                delta=0.1,       # 10%显著性阈值
                tau=0.15,        # 多臂老虎机参数
                batch_size=100,  # 批量大小
                coverage_samples=1000,  # 覆盖率样本数
                beam_size=2      # 候选锚点数量
            )
            
            # 显示锚点结果
            print(f"锚点规则: {' AND '.join(anchor_explanation.anchor)}")
            print(f"锚点精度: {anchor_explanation.precision:.3f}")
            print(f"锚点覆盖率: {anchor_explanation.coverage:.3f}")
            
            # 详细解释锚点规则
            print(f"\n锚点规则解释:")
            if len(anchor_explanation.anchor) == 0:
                print("  -> 空锚点：模型对该样本的预测不依赖于特定特征组合")
                print("  -> 这可能表明该样本的预测很稳定，或者位于决策边界之外")
            else:
                print("  -> 只要满足以下条件，模型预测就会保持不变：")
                for i, rule in enumerate(anchor_explanation.anchor):
                    print(f"     {i+1}. {rule}")
                
                print(f"\n  -> 该规则适用于 {anchor_explanation.coverage:.1%} 的人群")
                print(f"  -> 在满足条件的人群中，有 {anchor_explanation.precision:.1%} 的概率得到相同预测")
            
            # 显示锚点的额外信息
            if hasattr(anchor_explanation, 'raw') and anchor_explanation.raw:
                raw_data = anchor_explanation.raw
                if 'examples' in raw_data:
                    examples = raw_data['examples']
                    if len(examples['covered_true']) > 0:
                        print(f"\n满足锚点条件且预测正确的样本数: {len(examples['covered_true'])}")
                    if len(examples['covered_false']) > 0:
                        print(f"满足锚点条件但预测错误的样本数: {len(examples['covered_false'])}")
            
        except Exception as e:
            print(f"锚点解释出错: {str(e)}")
            anchor_explanation = None
        
        return {
            'sample_idx': sample_idx,
            'aplr_prediction': aplr_pred,
            'aplr_probability': aplr_prob,
            'true_label': sample_y,
            'aplr_explanation': aplr_local_explanation,
            'anchor_explanation': anchor_explanation
        }
    
    def explain_local_with_anchors(self, X, y, dataset_name, n_samples=3, specific_indices=None):
        """
        生成APLR局部解释和锚点解释
        """
        if self.model is None:
            print("请先训练模型")
            return None
        
        print(f"\n生成 {dataset_name} 数据集的局部解释和锚点解释...")
        
        # 设置锚点解释器
        if self.anchor_explainer is None:
            self.setup_anchor_explainer(X)
        
        # 选择要解释的样本
        if specific_indices is not None:
            sample_indices = specific_indices
            print(f"使用指定的样本索引: {sample_indices}")
        else:
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
            print(f"随机选择的样本索引: {sample_indices}")
        
        # 对每个样本进行详细解释
        all_explanations = []
        for sample_idx in sample_indices:
            explanation_result = self.explain_sample_with_anchors(X, y, sample_idx, dataset_name)
            all_explanations.append(explanation_result)
        
        # 生成总结
        print(f"\n{'='*80}")
        print(f"{dataset_name} 数据集解释总结")
        print(f"{'='*80}")
        
        for i, result in enumerate(all_explanations):
            print(f"\n样本 {i+1} (行索引: {result['sample_idx']}):")
            print(f"  真实/预测: {'ASD' if result['true_label'] == 1 else '无ASD'} / {'ASD' if result['aplr_prediction'] == 1 else '无ASD'}")
            print(f"  ASD概率: {result['aplr_probability'][1]:.3f}")
            
            if result['anchor_explanation'] is not None:
                anchor = result['anchor_explanation']
                if len(anchor.anchor) > 0:
                    print(f"  锚点规则: {' AND '.join(anchor.anchor)}")
                    print(f"  规则覆盖率: {anchor.coverage:.3f}")
                else:
                    print(f"  锚点规则: 空锚点（预测稳定）")
            else:
                print(f"  锚点规则: 解释失败")
        
        return {
            'explanations': all_explanations,
            'sample_indices': sample_indices,
            'dataset_name': dataset_name
        }
    
    def save_explanation_results(self, explanation_results, filename=None):
        """保存解释结果到Excel文件"""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            dataset_name = explanation_results['dataset_name']
            filename = f"{dataset_name}_explanations_{timestamp}.xlsx"
        
        # 准备保存的数据
        save_data = []
        for result in explanation_results['explanations']:
            row_data = {
                'sample_index': result['sample_idx'],
                'true_label': 'ASD' if result['true_label'] == 1 else '无ASD',
                'aplr_prediction': 'ASD' if result['aplr_prediction'] == 1 else '无ASD',
                'asd_probability': result['aplr_probability'][1],
                'prediction_correct': result['aplr_prediction'] == result['true_label']
            }
            
            # 添加锚点信息
            if result['anchor_explanation'] is not None:
                anchor = result['anchor_explanation']
                row_data.update({
                    'anchor_rules': ' AND '.join(anchor.anchor) if len(anchor.anchor) > 0 else '空锚点',
                    'anchor_precision': anchor.precision,
                    'anchor_coverage': anchor.coverage,
                    'anchor_rule_count': len(anchor.anchor)
                })
            else:
                row_data.update({
                    'anchor_rules': '解释失败',
                    'anchor_precision': None,
                    'anchor_coverage': None,
                    'anchor_rule_count': None
                })
            
            save_data.append(row_data)
        
        # 保存到Excel
        df = pd.DataFrame(save_data)
        df.to_excel(filename, index=False)
        print(f"解释结果已保存到: {filename}")
        
        return filename

def main():
    """主函数"""
    print("=== APLR + Anchors ASD预测分析系统 ===\n")
    
    # 分析L3-4数据集
    print("1. L3-4数据集分析")
    print("=" * 50)
    
    predictor_l34 = ASDPredictorWithAnchors()
    X_l34, y_l34 = predictor_l34.load_and_preprocess_data(DATA_PATH, L34_SHEET)
    model_l34 = predictor_l34.train_model(X_l34, y_l34)
    
    # 生成解释
    global_exp_l34 = predictor_l34.explain_global("L3-4")
    explanation_results_l34 = predictor_l34.explain_local_with_anchors(X_l34, y_l34, "L3-4", n_samples=3)
    
    # 保存结果
    # predictor_l34.save_explanation_results(explanation_results_l34)
    
    print("\n" + "="*80 + "\n")
    
    # 分析L5-S1数据集
    print("2. L5-S1数据集分析")
    print("=" * 50)
    
    predictor_l5s1 = ASDPredictorWithAnchors()
    X_l5s1, y_l5s1 = predictor_l5s1.load_and_preprocess_data(DATA_PATH, L5S1_SHEET)
    model_l5s1 = predictor_l5s1.train_model(X_l5s1, y_l5s1)
    
    # 生成解释
    global_exp_l5s1 = predictor_l5s1.explain_global("L5-S1")
    explanation_results_l5s1 = predictor_l5s1.explain_local_with_anchors(X_l5s1, y_l5s1, "L5-S1", n_samples=3)
    
    # 保存结果
    # predictor_l5s1.save_explanation_results(explanation_results_l5s1)
    
    print("\n=== 分析完成 ===")
    
    return {
        'l34': {
            'predictor': predictor_l34,
            'X': X_l34,
            'y': y_l34,
            'explanations': explanation_results_l34
        },
        'l5s1': {
            'predictor': predictor_l5s1,
            'X': X_l5s1,
            'y': y_l5s1,
            'explanations': explanation_results_l5s1
        }
    }

# 使用示例
if __name__ == "__main__":
    results = main()
    
    # 如果需要对特定样本进行解释，可以这样做：
    # specific_results = results['l34']['predictor'].explain_local_with_anchors(
    #     results['l34']['X'], 
    #     results['l34']['y'], 
    #     "L3-4", 
    #     specific_indices=[0, 10, 20]
    # )


# In[ ]:





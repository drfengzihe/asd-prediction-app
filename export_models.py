#!/usr/bin/env python
# coding: utf-8

"""
模型和数据导出脚本
将训练好的模型和解释结果导出为Streamlit应用所需的格式
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.dummy import DummyClassifier  # 用于演示，替换为你的实际模型


def create_directories():
    """创建必要的目录"""
    directories = ['models', 'data', 'utils']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")


def export_demo_models():
    """导出演示模型（你需要替换为实际训练的模型）"""
    print("导出演示模型...")

    # 创建虚拟特征名称（基于你的研究）
    l34_features = [
        'Gender', 'Age', 'BMI', 'Hypertension', 'Diabetes', 'Smoking history',
        'Alcohol abuse', 'L3-4 pfirrmann grade', 'L3-4 spinal canal stenosis',
        'L3-4 foraminal stenosis', 'L3-4 modic change', 'L3-4 osteoarthritis of facet joints',
        'L3-4 sagittal imbalance', 'L3-4 coronal imbalance', 'L3-4 local lordosis angle',
        'L3-4 EBQ', 'L3-4 preoperative disc height', 'L5-S1 pfirrmann grade',
        'L5-S1 spinal canal stenosis', 'L5-S1 foraminal stenosis', 'L5-S1 EBQ',
        'L5-S1 local lordosis angle', 'Hounsfield Units', 'Cage height',
        'Operative time', 'Blood loss'
    ]

    l5s1_features = l34_features.copy()  # 假设特征相同

    # 创建演示模型（替换为你的实际APLR模型）
    np.random.seed(42)

    # 生成虚拟训练数据
    n_samples = 100
    l34_X = np.random.randn(n_samples, len(l34_features))
    l34_y = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    l5s1_X = np.random.randn(n_samples, len(l5s1_features))
    l5s1_y = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

    # 训练虚拟模型
    l34_model = DummyClassifier(strategy='stratified', random_state=42)
    l34_model.fit(l34_X, l34_y)

    l5s1_model = DummyClassifier(strategy='stratified', random_state=42)
    l5s1_model.fit(l5s1_X, l5s1_y)

    # 保存模型
    with open('models/l34_model.pkl', 'wb') as f:
        pickle.dump(l34_model, f)

    with open('models/l5s1_model.pkl', 'wb') as f:
        pickle.dump(l5s1_model, f)

    # 保存特征信息
    feature_info = {
        'l34': {
            'feature_names': l34_features,
            'feature_count': len(l34_features)
        },
        'l5s1': {
            'feature_names': l5s1_features,
            'feature_count': len(l5s1_features)
        }
    }

    with open('models/feature_info.json', 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    print("✅ 模型导出完成")


def export_demo_explanations():
    """导出演示解释结果"""
    print("导出演示解释结果...")

    # 全局解释（特征重要性）
    global_explanations = {
        'l34': {
            'feature_names': ['L3-4 EBQ', 'L5-S1 EBQ', 'Age', 'L3-4 pfirrmann grade',
                              'Facet osteoarthritis', 'L3-4 local lordosis', 'BMI',
                              'Foraminal stenosis', 'Modic changes', 'Cage height'],
            'importances': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01],
            'metrics': {
                'accuracy': 0.988,
                'roc_auc': 0.999,
                'recall': 0.952,
                'precision': 0.965
            }
        },
        'l5s1': {
            'feature_names': ['Facet osteoarthritis', 'L3-4 EBQ', 'L5-S1 local lordosis',
                              'L5-S1 EBQ', 'Age', 'Foraminal stenosis', 'Modic changes',
                              'BMI', 'Operative time', 'Blood loss'],
            'importances': [0.22, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03],
            'metrics': {
                'accuracy': 0.886,
                'roc_auc': 0.914,
                'recall': 0.611,
                'precision': 0.742
            }
        }
    }

    with open('data/global_explanations.json', 'w', encoding='utf-8') as f:
        json.dump(global_explanations, f, ensure_ascii=False, indent=2)

    # 局部解释示例
    local_explanations = {
        'l34': [
            {
                'patient_id': 1,
                'true_label': 1,
                'prediction': 1,
                'probability': 0.952,
                'features': ['L3-4 EBQ', 'L5-S1 EBQ', 'Age', 'L3-4 lordosis'],
                'contributions': [0.35, 0.22, 0.15, -0.08]
            },
            {
                'patient_id': 2,
                'true_label': 0,
                'prediction': 0,
                'probability': 0.123,
                'features': ['L3-4 EBQ', 'Facet osteoarthritis', 'BMI', 'Age'],
                'contributions': [-0.25, -0.18, -0.12, 0.08]
            }
        ],
        'l5s1': [
            {
                'patient_id': 1,
                'true_label': 1,
                'prediction': 1,
                'probability': 0.742,
                'features': ['Facet osteoarthritis', 'L3-4 EBQ', 'L5-S1 lordosis', 'Age'],
                'contributions': [0.28, 0.20, 0.15, 0.10]
            },
            {
                'patient_id': 2,
                'true_label': 0,
                'prediction': 0,
                'probability': 0.234,
                'features': ['Facet osteoarthritis', 'L5-S1 EBQ', 'Modic changes', 'BMI'],
                'contributions': [-0.22, -0.16, -0.10, 0.05]
            }
        ]
    }

    with open('data/local_explanations.json', 'w', encoding='utf-8') as f:
        json.dump(local_explanations, f, ensure_ascii=False, indent=2)

    # 锚点解释示例
    anchor_examples = {
        'l34': [
            {
                'patient_id': 1,
                'prediction': 1,
                'rules': [
                    'L3-4 EBQ > 3.60',
                    'L5-S1 EBQ ≤ 3.50',
                    'L3-4 pfirrmann grade > 4.00'
                ],
                'precision': 1.000,
                'coverage': 0.043
            },
            {
                'patient_id': 2,
                'prediction': 0,
                'rules': [
                    'L3-4 EBQ ≤ 2.50',
                    'Age ≤ 55'
                ],
                'precision': 0.920,
                'coverage': 0.125
            }
        ],
        'l5s1': [
            {
                'patient_id': 1,
                'prediction': 1,
                'rules': [
                    'L5-S1 facet osteoarthritis > 1.00',
                    'L3-4 EBQ > 3.60',
                    'L5-S1 local lordosis ≤ 11.70'
                ],
                'precision': 0.951,
                'coverage': 0.038
            }
        ]
    }

    with open('data/anchor_examples.json', 'w', encoding='utf-8') as f:
        json.dump(anchor_examples, f, ensure_ascii=False, indent=2)

    print("✅ 解释结果导出完成")


def export_real_models_template():
    """为导出真实模型提供模板代码"""
    template_code = '''
# 导出真实训练模型的模板代码
# 请根据你的实际代码修改以下部分

def export_real_aplr_models():
    """导出真实的APLR模型"""

    # 1. 加载你训练好的模型
    # predictor_l34 = your_trained_l34_predictor
    # predictor_l5s1 = your_trained_l5s1_predictor

    # 2. 提取模型
    # l34_model = predictor_l34.model
    # l5s1_model = predictor_l5s1.model

    # 3. 保存模型
    # with open('models/l34_model.pkl', 'wb') as f:
    #     pickle.dump(l34_model, f)
    # with open('models/l5s1_model.pkl', 'wb') as f:
    #     pickle.dump(l5s1_model, f)

    # 4. 导出全局解释
    # l34_global = predictor_l34.explain_global("L3-4")
    # l5s1_global = predictor_l5s1.explain_global("L5-S1")

    # 5. 提取特征重要性等信息并保存为JSON

    pass

# 使用方法:
# 1. 运行你的原始训练代码得到 predictor_l34 和 predictor_l5s1
# 2. 调用上面的函数导出模型
# 3. 运行 export_demo_explanations() 导出解释结果
'''

    with open('export_real_models_template.py', 'w', encoding='utf-8') as f:
        f.write(template_code)

    print("✅ 真实模型导出模板已创建: export_real_models_template.py")


def main():
    """主函数"""
    print("🚀 开始导出模型和数据...")

    # 创建目录
    create_directories()

    # 导出演示模型（你需要替换为真实模型）
    export_demo_models()

    # 导出演示解释结果
    export_demo_explanations()

    # 创建真实模型导出模板
    export_real_models_template()

    print("\n✅ 导出完成！")
    print("\n📝 接下来的步骤:")
    print("1. 运行你的原始训练代码")
    print("2. 使用 export_real_models_template.py 中的模板导出真实模型")
    print("3. 运行 streamlit run app.py 启动应用")


if __name__ == "__main__":
    main()
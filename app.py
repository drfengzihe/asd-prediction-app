# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from alibi.explainers import AnchorTabular
import warnings

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="ASD预测分析系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ASDPredictor:
    """ASD预测器类"""

    def __init__(self):
        self.aplr_models = None
        self.scalers = None
        self.feature_names = None
        self.categorical_features = None
        self.sample_data = None
        self.anchor_explainers = {}

    def load_models(self):
        """加载预训练的模型"""
        try:
            with open('models/aplr_models.pkl', 'rb') as f:
                self.aplr_models = pickle.load(f)

            with open('models/scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)

            with open('models/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)

            with open('models/categorical_features.pkl', 'rb') as f:
                self.categorical_features = pickle.load(f)

            with open('models/sample_data.pkl', 'rb') as f:
                self.sample_data = pickle.load(f)

            return True
        except FileNotFoundError as e:
            st.error(f"模型文件未找到: {str(e)}")
            return False

    def setup_anchor_explainer(self, dataset_key):
        """设置锚点解释器"""
        if dataset_key in self.anchor_explainers:
            return self.anchor_explainers[dataset_key]

        try:
            # 创建预测函数
            model = self.aplr_models[dataset_key]
            feature_names = self.feature_names[dataset_key]

            def predict_fn(x):
                if isinstance(x, np.ndarray):
                    x_df = pd.DataFrame(x, columns=feature_names)
                else:
                    x_df = x
                predictions = model.predict(x_df)
                return np.array([int(p) for p in predictions])

            # 创建锚点解释器
            anchor_explainer = AnchorTabular(
                predictor=predict_fn,
                feature_names=feature_names,
                categorical_names=self.categorical_features[dataset_key] if self.categorical_features[
                    dataset_key] else None
            )

            # 使用保存的样本数据拟合
            X_sample = self.sample_data[dataset_key]['X_sample']
            anchor_explainer.fit(X_sample, disc_perc=[25, 50, 75])

            self.anchor_explainers[dataset_key] = anchor_explainer
            return anchor_explainer

        except Exception as e:
            st.warning(f"锚点解释器设置失败: {str(e)}")
            return None

    def predict_sample(self, dataset_key, input_data):
        """对新样本进行预测并生成解释"""
        try:
            model = self.aplr_models[dataset_key]
            feature_names = self.feature_names[dataset_key]

            # 创建DataFrame
            input_df = pd.DataFrame([input_data], columns=feature_names)

            # APLR预测
            prediction_raw = model.predict(input_df)
            prediction = int(prediction_raw[0])
            probabilities = model.predict_class_probabilities(input_df)[0]

            # 锚点解释
            anchor_exp = None
            anchor_explainer = self.setup_anchor_explainer(dataset_key)

            if anchor_explainer is not None:
                try:
                    anchor_exp = anchor_explainer.explain(
                        input_df.values[0],
                        threshold=0.85,
                        delta=0.1,
                        tau=0.15,
                        batch_size=100,
                        coverage_samples=500,
                        beam_size=2
                    )
                except Exception as e:
                    st.warning(f"锚点解释生成失败: {str(e)}")

            return prediction, probabilities, anchor_exp

        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")
            return None, None, None


@st.cache_resource
def load_predictor():
    """缓存加载预测器"""
    predictor = ASDPredictor()
    if predictor.load_models():
        return predictor
    return None


def create_probability_chart(probabilities):
    """创建概率图表"""
    prob_asd = probabilities[1]
    prob_no_asd = probabilities[0]

    fig = go.Figure(data=[
        go.Bar(
            x=['不发生ASD', '发生ASD'],
            y=[prob_no_asd, prob_asd],
            marker_color=['green' if prob_no_asd > prob_asd else 'lightgreen',
                          'red' if prob_asd > prob_no_asd else 'lightcoral'],
            text=[f'{prob_no_asd:.1%}', f'{prob_asd:.1%}'],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="预测概率分布",
        yaxis_title="概率",
        showlegend=False,
        height=400
    )
    return fig


def display_research_results(predictor, dataset_key, dataset_name):
    """显示研究结果"""
    st.header(f"📊 {dataset_name} 数据集研究结果")

    sample_data = predictor.sample_data[dataset_key]

    # 显示数据集基本信息
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("样本数量", sample_data['sample_size'])

    with col2:
        asd_count = sample_data['target_distribution'].get(1, 0)
        st.metric("ASD患者数", asd_count)

    with col3:
        asd_rate = asd_count / sample_data['sample_size'] * 100
        st.metric("ASD发生率", f"{asd_rate:.1f}%")

    # 特征统计信息
    st.subheader("📈 重要特征统计信息")

    stats_df = pd.DataFrame(sample_data['feature_stats'])

    # 显示部分重要特征的统计信息
    important_features = ['Age', 'BMI', 'L3-4 EBQ', 'L5-S1 EBQ', 'L3-4 pfirrmann grade', 'L5-S1 pfirrmann grade']
    available_features = [f for f in important_features if f in stats_df.columns]

    if available_features:
        display_stats = stats_df[available_features].round(2)
        st.dataframe(display_stats, use_container_width=True)

    # 模型性能信息
    st.subheader("🎯 模型信息")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **✅ APLR模型**
        - 模型类型: 自动分段线性回归
        - 特征数量: {len(predictor.feature_names[dataset_key])}
        - 训练状态: 已完成
        """)

    with col2:
        st.info(f"""
        **⚓ 解释能力**
        - 全局解释: 支持
        - 局部解释: 支持
        - 锚点解释: 支持
        """)

    # 显示特征列表
    with st.expander("📋 查看所有特征列表"):
        feature_df = pd.DataFrame({
            '序号': range(1, len(predictor.feature_names[dataset_key]) + 1),
            '特征名称': predictor.feature_names[dataset_key]
        })
        st.dataframe(feature_df, use_container_width=True)


def display_prediction_interface(predictor, dataset_key, dataset_name):
    """显示预测界面"""
    st.header(f"🔮 {dataset_name} 新样本预测")

    st.markdown("""
    请输入患者的特征值，系统将使用训练好的APLR模型进行ASD风险预测，
    并提供锚点解释分析。
    """)

    # 创建输入表单
    with st.form("prediction_form"):
        st.subheader("患者特征输入")

        input_data = {}
        features = predictor.feature_names[dataset_key]
        stats = predictor.sample_data[dataset_key]['feature_stats']

        # 分成多列显示
        cols = st.columns(3)
        col_idx = 0

        for feature in features:
            with cols[col_idx % 3]:
                # 根据特征名称判断输入类型
                if feature.lower() in ['gender']:
                    input_data[feature] = st.selectbox(
                        feature,
                        options=[0, 1],
                        format_func=lambda x: '女性' if x == 0 else '男性',
                        key=f"{dataset_key}_{feature}"
                    )
                elif feature.lower() in ['hypertension', 'diabetes', 'smoking history', 'alcohol abuse']:
                    input_data[feature] = st.selectbox(
                        feature,
                        options=[0, 1],
                        format_func=lambda x: '否' if x == 0 else '是',
                        key=f"{dataset_key}_{feature}"
                    )
                elif any(word in feature.lower() for word in
                         ['pfirrmann', 'grade', 'stenosis', 'modic', 'osteoarthritis']):
                    min_val = int(stats['min'][feature])
                    max_val = int(stats['max'][feature])
                    median_val = int(stats['median'][feature])
                    input_data[feature] = st.slider(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        key=f"{dataset_key}_{feature}"
                    )
                else:
                    # 数值型特征
                    min_val = float(stats['min'][feature])
                    max_val = float(stats['max'][feature])
                    mean_val = float(stats['mean'][feature])

                    input_data[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=0.1 if max_val - min_val < 100 else 1.0,
                        key=f"{dataset_key}_{feature}"
                    )

            col_idx += 1

        submitted = st.form_submit_button("🔍 开始预测", use_container_width=True)

    if submitted:
        with st.spinner("正在进行预测和生成解释..."):
            # 进行预测
            prediction, probabilities, anchor_exp = predictor.predict_sample(
                dataset_key, list(input_data.values())
            )

            if prediction is not None:
                # 显示预测结果
                st.success("✅ 预测完成！")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🎯 预测结果")

                    if prediction == 1:
                        st.error("⚠️ **高风险**：预测发生ASD")
                    else:
                        st.success("✅ **低风险**：预测不发生ASD")

                    # 概率显示
                    prob_asd = probabilities[1]
                    prob_no_asd = probabilities[0]

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("ASD发生概率", f"{prob_asd:.1%}")
                    with col_b:
                        st.metric("不发生ASD概率", f"{prob_no_asd:.1%}")

                    # 概率图表
                    fig_prob = create_probability_chart(probabilities)
                    st.plotly_chart(fig_prob, use_container_width=True)

                with col2:
                    st.subheader("⚓ 锚点解释")

                    if anchor_exp is not None:
                        if len(anchor_exp.anchor) == 0:
                            st.info("🔄 **空锚点**：模型对该样本的预测非常稳定")
                            st.markdown("""
                            **含义：** 无论其他特征如何微调，预测结果都不会改变。
                            这通常表明该患者的风险模式非常明确。
                            """)
                        else:
                            st.write("**🔑 关键决策规则：**")
                            for i, rule in enumerate(anchor_exp.anchor):
                                st.markdown(f"**{i + 1}.** `{rule}`")

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("规则精确度", f"{anchor_exp.precision:.1%}")
                            with col_b:
                                st.metric("规则覆盖率", f"{anchor_exp.coverage:.1%}")

                            # 质量评估
                            st.markdown("**📊 规则质量评估：**")
                            if anchor_exp.precision >= 0.9:
                                st.success("✅ 精确度：优秀 (≥90%)")
                            elif anchor_exp.precision >= 0.8:
                                st.warning("⚠️ 精确度：良好 (80-90%)")
                            else:
                                st.error("❌ 精确度：需要改进 (<80%)")

                            st.markdown(f"""
                            **💡 解释：** 当患者同时满足上述 {len(anchor_exp.anchor)} 个条件时，
                            模型有 {anchor_exp.precision:.1%} 的把握预测为 {'ASD' if prediction == 1 else '无ASD'}。
                            这种特征组合出现在约 {anchor_exp.coverage:.1%} 的患者中。
                            """)
                    else:
                        st.warning("⚠️ 无法生成锚点解释")

                # 特征值总结
                st.subheader("📋 输入特征总结")

                # 创建特征值表格
                feature_summary = []
                for feature, value in input_data.items():
                    mean_val = stats['mean'][feature]
                    diff = value - mean_val

                    if abs(diff) < 0.01:
                        comparison = "≈ 平均值"
                        color = "🟡"
                    elif diff > 0:
                        comparison = f"高于平均 (+{diff:.2f})"
                        color = "🔴" if diff > stats['std'][feature] else "🟠"
                    else:
                        comparison = f"低于平均 ({diff:.2f})"
                        color = "🔵" if abs(diff) > stats['std'][feature] else "🟢"

                    feature_summary.append({
                        '特征名称': feature,
                        '输入值': f"{value:.2f}" if isinstance(value, float) else str(value),
                        '平均值': f"{mean_val:.2f}",
                        '与平均值比较': f"{color} {comparison}"
                    })

                summary_df = pd.DataFrame(feature_summary)
                st.dataframe(summary_df, use_container_width=True)

                # 医学建议
                st.subheader("💡 个性化医学建议")

                if prediction == 1:
                    st.markdown("""
                    <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
                    <h4 style="color: #d32f2f;">🚨 高风险患者管理建议</h4>
                    <ul>
                    <li>🔍 <strong>密切监控：</strong> 建议3-6个月进行影像学复查</li>
                    <li>🏃‍♂️ <strong>功能锻炼：</strong> 加强核心肌群和腰背肌训练</li>
                    <li>⚖️ <strong>体重管理：</strong> 控制BMI，减少脊柱负荷</li>
                    <li>🚭 <strong>生活方式：</strong> 戒烟限酒，改善骨质健康</li>
                    <li>💊 <strong>药物干预：</strong> 考虑抗骨质疏松治疗</li>
                    <li>🩺 <strong>随访计划：</strong> 制定个性化长期随访方案</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50;">
                    <h4 style="color: #388e3c;">✅ 低风险患者维护建议</h4>
                    <ul>
                    <li>✅ <strong>保持现状：</strong> 继续维持良好的生活习惯</li>
                    <li>🏃‍♂️ <strong>适度运动：</strong> 规律进行有氧和力量训练</li>
                    <li>📅 <strong>常规随访：</strong> 建议6-12个月复查一次</li>
                    <li>⚠️ <strong>症状监控：</strong> 注意腰背部不适，及时就医</li>
                    <li>🥗 <strong>营养补充：</strong> 保证钙质和维生素D摄入</li>
                    <li>😊 <strong>心理健康：</strong> 保持积极乐观的心态</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)


def main():
    # 加载预测器
    predictor = load_predictor()

    if predictor is None:
        st.error("❌ 模型加载失败！请确保模型文件存在。")
        st.stop()

    # 标题和介绍
    st.title("🏥 ASD预测分析系统")
    st.markdown("""
    **相邻节段病变(Adjacent Segment Disease, ASD)预测系统**

    基于自动分段线性回归(APLR)模型和锚点解释技术，为脊柱融合术后患者提供个性化风险评估。
    """)
    st.markdown("---")

    # 侧边栏导航
    st.sidebar.title("🧭 系统导航")
    st.sidebar.markdown("---")

    # 选择分析类型
    analysis_type = st.sidebar.selectbox(
        "📊 选择功能模块",
        ["研究结果展示", "新样本预测"],
        help="选择要使用的功能模块"
    )

    # 选择数据集
    dataset = st.sidebar.selectbox(
        "🎯 选择预测目标",
        ["L3-4", "L5-S1"],
        format_func=lambda x: f"{x} 节段ASD预测",
        help="选择要分析的相邻节段"
    )

    dataset_key = 'l34' if dataset == 'L3-4' else 'l5s1'

    # 显示系统信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 系统信息")
    st.sidebar.info(f"""
    **当前配置：**
    - 目标节段: {dataset}
    - 功能模块: {analysis_type}
    - 模型状态: ✅ 已加载
    - 解释功能: ✅ 可用
    """)

    # 根据选择显示相应功能
    if analysis_type == "研究结果展示":
        display_research_results(predictor, dataset_key, dataset)
    elif analysis_type == "新样本预测":
        display_prediction_interface(predictor, dataset_key, dataset)

    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
    <p>🔬 基于可解释AI技术 | ⚕️ 仅供医学研究参考 | 🏥 请结合临床实际情况</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
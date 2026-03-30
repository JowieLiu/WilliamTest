import streamlit as st
import requests

st.set_page_config(page_title="AAPL 10-K 财报问答系统", page_icon="📊")

st.title("📊 Apple 10-K 财报智能问答系统")

st.markdown("""
基于 Apple 2020-2025 年 10-K 报告的智能金融问答系统，使用 RAG 技术实现。
""")

# 侧边栏：模型配置
with st.sidebar:
    st.header("⚙️ 模型配置")
    
    # 显示当前使用的模型
    try:
        info_response = requests.get("http://localhost:8000/api/generator-info")
        if info_response.status_code == 200:
            generator_info = info_response.json()
            st.success(f"✅ 当前使用: {generator_info['name']}")
            st.info(generator_info['description'])
            
            if generator_info['type'] == 'api':
                st.caption(f"API URL: {generator_info.get('api_url', '')}")
                st.caption(f"模型名称: {generator_info.get('model_name', '')}")
    except:
        st.warning("⚠️ 无法连接到后端服务")
    
    st.divider()
    
    st.subheader("配置 API 大模型")
    st.caption("如果需要使用大模型，请填写以下信息：")
    
    api_url = st.text_input(
        "API URL",
        placeholder="例如: https://api.openai.com/v1"
    )
    
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="输入您的 API Key"
    )
    
    model_name = st.text_input(
        "模型名称",
        value="gpt-3.5-turbo",
        placeholder="例如: gpt-3.5-turbo, gpt-4"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("使用 API 大模型", type="primary"):
            if api_url and api_key:
                try:
                    config_response = requests.post(
                        "http://localhost:8000/api/configure-llm",
                        json={
                            "api_url": api_url,
                            "api_key": api_key,
                            "model_name": model_name
                        }
                    )
                    if config_response.status_code == 200:
                        st.success("✅ 配置成功！")
                        st.rerun()
                    else:
                        st.error("配置失败")
                except Exception as e:
                    st.error(f"配置错误: {str(e)}")
            else:
                st.warning("请填写 API URL 和 API Key")
    
    with col2:
        if st.button("使用本地小模型"):
            try:
                config_response = requests.post(
                    "http://localhost:8000/api/configure-llm",
                    json={"api_url": None, "api_key": None}
                )
                if config_response.status_code == 200:
                    st.success("✅ 已切换到本地小模型")
                    st.rerun()
                else:
                    st.error("切换失败")
            except Exception as e:
                st.error(f"切换错误: {str(e)}")
    
    st.divider()
    st.info("💡 提示：不填写 API 配置时，系统将使用本地 TinyLlama 小模型进行总结。")

# 主界面：问答功能
question = st.text_input(
    "请输入您的问题：",
    placeholder="例如：Apple 2025 年的营收是多少？"
)

top_k = st.slider("检索相关文档数量", min_value=1, max_value=5, value=3)

if st.button("获取回答"):
    if question:
        with st.spinner("正在思考..."):
            try:
                response = requests.post(
                    "http://localhost:8000/api/qa",
                    json={"question": question, "top_k": top_k}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("💡 回答：")
                    st.write(result["answer"])
                    
                    st.subheader("📚 参考来源：")
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(f"来源 {i} - {source['metadata'].get('year', 'N/A')} 年 {source['metadata'].get('section_title', 'N/A')}"):
                            st.write(source["text"])
                            if source.get('distance'):
                                st.caption(f"相似度距离: {source['distance']:.4f}")
                else:
                    st.error(f"错误: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("无法连接到后端服务，请确保后端服务正在运行！")
                st.info("请先运行 `python main.py` 启动后端服务")
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
    else:
        st.warning("请输入问题！")

st.markdown("---")
st.markdown("### 示例问题：")
example_questions = [
    "Apple 2025 年的主要产品有哪些？",
    "Apple 的风险因素有哪些？",
    "2025 年 Apple 的营收是多少？",
    "Apple 的管理层讨论与分析中提到了哪些市场风险？"
]
for q in example_questions:
    if st.button(q, key=q):
        st.session_state['question'] = q
        st.rerun()

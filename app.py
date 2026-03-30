import streamlit as st
import requests

st.set_page_config(page_title="AAPL 10-K 财报问答系统", page_icon="📊")

st.title("📊 Apple 10-K 财报智能问答系统")

st.markdown("""
基于 Apple 2020-2025 年 10-K 报告的智能金融问答系统，使用 RAG 技术实现。
""")

question = st.text_input(
    "请输入您的问题：",
    placeholder="例如：Apple 2025 年的营收是多少？"
)

top_k = st.slider("检索相关文档数量", min_value=1, max_value=10, value=5)

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

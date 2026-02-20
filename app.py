import streamlit as st
import asyncio
from NovaCortexbot import YarmoukAgent, init_db # تأكد أن اسم ملفك NovaCortexbot.py

# إعدادات الصفحة
st.set_page_config(page_title="NovaCortex - جامعة اليرموك", page_icon="🏛️")

# تهيئة قاعدة البيانات والعميل (مرة واحدة فقط)
if 'agent' not in st.session_state:
    init_db()
    st.session_state.agent = YarmoukAgent()
    st.session_state.messages = []

st.title("🏛️ NovaCortex v12.9.2")
st.markdown("### المساعد الذكي لجامعة اليرموك (نسخة الويب)")

# عرض سجل المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# استقبال السؤال من المستخدم
if prompt := st.chat_input("كيف يمكنني مساعدتك بخصوص جامعة اليرموك؟"):
    # إضافة سؤال المستخدم للسجل
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # معالجة الرد
    with st.chat_message("assistant"):
        with st.spinner("جاري استخراج البيانات..."):
            # تشغيل الدالة async داخل Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # نستخدم معرف ثابت للمتصفح أو session_id كـ user_id
            response = loop.run_until_complete(st.session_state.agent.ask(prompt, user_id=12345))
            
            st.markdown(response["response"])
            st.session_state.messages.append({"role": "assistant", "content": response["response"]})
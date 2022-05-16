import streamlit as st
st.markdown(""" div.stButton > button:first-child {
background-color: #00cc00;color:white;font-size:20px;height:3em;width:30em;border-radius:10px 10px 10px 10px;
}
""", unsafe_allow_html=True)

if st.button("the notice you want to show"):
	st.write("content you want to show")
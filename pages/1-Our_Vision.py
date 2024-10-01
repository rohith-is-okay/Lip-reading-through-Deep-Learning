import streamlit as st
# Define CSS styles
main_bg_color = "#fffff"
main_txt_color = "#000000"
accent_color = "#00001a"

# Apply CSS styles
st.markdown(
    f"""
    <style>
        
    
        .sidebar .sidebar-content {{
            background-color: {accent_color};
            color: {main_txt_color};
        }}
        .sidebar .sidebar-content .block-container {{
            background-color: transparent;
        }}
        .Widget>label {{
            color: {main_txt_color};
        }}
        .st-bw {{
            background-color: {main_bg_color};
        }}
        .st-c3 .css-1v1l6e3 {{
            background-color: {accent_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar content
with st.sidebar: 
    st.image("https://cdn.pixabay.com/photo/2013/07/12/18/17/equalizer-153212_1280.png", width=150)
    st.title('LipSync Studio')
    st.info('This is made by using Deep Learning with help of CNN and RNN algorithms.')
    st.info('PARTH & ROHITH')

st.markdown('<h1 class="blue-header">Our Vision</h1>', unsafe_allow_html=True)
st.markdown('<hr class="blue-divider">', unsafe_allow_html=True)
st.write("""- Utilizing Deep Learning to Overcome Communication Barriers and redefine AVSR.
- Cost-effective Solution: Our vision is to provide a cost-effective alternative to traditional hearing aids, utilizing existing smartphone technology.
- Deep Learning Integration: By leveraging deep learning technology, we aim to develop a precise lip-reading model capable of interpreting visual speech cues.
- Our project envisions a future where individuals with hearing impairments can communicate effectively and independently.""")
st.markdown('<h1 class="blue-header">Objective</h1>', unsafe_allow_html=True)
st.markdown('<hr class="blue-divider">', unsafe_allow_html=True)
st.write("""- Develop a model to accurately translate complete sentences based on lip movements.
- Aim to achieve higher accuracy and dependability than human lip-readers through model training.
- Establish Cost-Free Communication Solution which is comparatively faster
- Create an application that is compatible with various devices, ensuring accessibility for all users.
- Develop the model to retain its accuracy in challenging environments""")




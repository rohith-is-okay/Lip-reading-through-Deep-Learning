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


st.markdown('<h1 class="blue-header">Introduction</h1>', unsafe_allow_html=True)
st.markdown('<hr class="blue-divider">', unsafe_allow_html=True)
st.write("""- Over 430 million people, including 34 million children, currently experience disabling hearing loss, a number projected to exceed 700 million by 2050.
- Hearing impairments often lead to social isolation and hinder success in education and careers due to limited communication options.
- Traditional hearing aids are available, but they cost around Rs 30,000 to 7 Lakh and up to 5000 dollars in abroad.
- Leveraging deep learning, our model empowers individuals with hearing impairments through accurate lip-reading.""")

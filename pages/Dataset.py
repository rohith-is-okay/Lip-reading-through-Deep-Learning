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
st.header('Dataset', divider='blue')


text = "We employ the GRID corpus for training due to its sentence-level nature and extensive dataset. " \
       "The sentences are structured using a straightforward grammar pattern: command(4) + color(4) + " \
       "preposition(4) + letter(25) + digit(10) + adverb(4). Each number indicates the available word " \
       "choices within the corresponding word category. These categories encompass {bin, lay, place, set}, " \
       "{blue, green, red, white}, {at, by, in, with}, {A,...,Z}{W}, {zero, ..., nine}, and " \
       "{again, now, please, soon}, resulting in a pool of 64,000 potential sentences. For instance, " \
       "exemplars from this dataset include 'set blue by A four please' and 'place red at C zero again'. " \
       "Model training was conducted on the first 450 videos of speaker one."

st.write(text)



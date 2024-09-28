import streamlit as st
from streamlit_option_menu import option_menu
from app import main
from scraper import run
from pdf_chatbot import chat_pdf
from data_analysis import anal

st.set_page_config(layout="wide")
selected = option_menu(
    menu_title=None,
    options=['Chatbot', 'PDF Chatbot', 'Research Papers', 'Data Analysis'],
    icons=['robot', 'file-earmark-pdf', 'book', 'graph-up'],
    menu_icon='cast',
    default_index=0,
    orientation='horizontal',
    )

if selected == 'Chatbot':
    main()
if selected == 'PDF Chatbot':
    chat_pdf()
if selected == 'Research Papers':
    run()
if selected == 'Data Analysis':
    anal()
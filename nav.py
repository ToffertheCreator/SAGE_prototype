# SAGE (Scholarly Assistant Generative Engine) - Your personal AI research assistant 
# Copyright (C) 2024 Kristopher Molina
#
# Licensed under the Apache License, Version 2.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, the software
# distributed under the License is provided on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This project provides research assistance with features like literature search, AI based research chatbot, 
# multimodal file input (e.g., pdf files, images, audio), while also supporting machine learning training and exploratory data analysis (EDA).
# It is an open-source project, allowing for contributions and modifications under the terms 
# of the License. The authors do not assume liability for any errors or omissions in the code.


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

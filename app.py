import streamlit as st
from llm_chains import load_normal_chain
from streamlit_mic_recorder import mic_recorder
from audio_handler import transcribe_audio
from image_handler import handle_image
#from pdf_handler import add_documents_to_db
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils import save_chat_history_json, get_timestamp, load_chat_history_json
import yaml
import os

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Function to load the appropriate chain based on the chat mode
def load_chain(chat_history):
    return load_normal_chain(chat_history)

# Helper function to clear the input field
def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ''

# Function to set the input state for sending
def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

# Track the session index
def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

# Function to save chat history
def save_chat_history():
    if st.session_state.history:
        if st.session_state.session_key == 'new_session':
            st.session_state.new_session_key = get_timestamp() + '.json'
            file_name = st.session_state.new_session_key
        else:
            file_name = st.session_state.session_key

        file_path = os.path.join(config['chat_history_path'], file_name)
        save_chat_history_json(st.session_state.history, file_path)


# Main function to run the app
def main():
    st.title('SAGE')
    chat_container = st.container()
    st.sidebar.title('Chat Sessions')

    # Load chat sessions from file
    chat_sessions = ['new_session'] + [f for f in os.listdir(config['chat_history_path']) if f.endswith('.json')]

    # Initialize session state variables if not already initialized
    if 'send_input' not in st.session_state:
        st.session_state.send_input = False
        st.session_state.user_question = ''
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = 'new_session'
    
    if 'session_key' not in st.session_state:
        st.session_state.session_key = 'new_session'

    # Initialize the `pdf_chat` session state variable if not already initialized
    if 'pdf_chat' not in st.session_state:
        st.session_state.pdf_chat = False
    
    if 'session_key_widget' not in st.session_state:
        st.session_state.session_key_widget = 'new_session'
    
    if st.session_state.session_key_widget != st.session_state.session_key:
        st.session_state.session_key = st.session_state.session_key_widget
    
    # Sidebar session selection
    st.sidebar.selectbox(
        'Select a chat session', 
        chat_sessions, 
        key='session_key_widget',  # Use separate key for widget
        index=chat_sessions.index(st.session_state.session_key)
    )

    # Session tracking logic
    if st.session_state.session_key == 'new_session' and st.session_state.new_session_key:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    if st.session_state.session_key == 'new_session' and st.session_state.new_session_key is not None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    # Update session index tracker
    if st.session_state.session_index_tracker in chat_sessions:
        index = chat_sessions.index(st.session_state.session_index_tracker)
    else:
        index = 0  # Default to the first session if not found
    #st.sidebar.selectbox('Select a chat session', chat_sessions, key='session_key', index=index, on_change=track_index)

    # Load chat history if session is not new
    if st.session_state.session_key != 'new_session':
        file_path = os.path.join(config['chat_history_path'], st.session_state.session_key)
        st.session_state.history = load_chat_history_json(file_path)
    else:
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key='history')
    try:
        llm_chain = load_chain(chat_history)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return

    # Input field and button
    user_input = st.text_input('Type your message here', key='user_input', on_change=set_send_input)
    voice_recording_column, send_button_column = st.columns(2)
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt='Start recording', stop_prompt='Stop recording', just_once=True)
    with send_button_column:
        send_button = st.button('Send', key='send_button', on_click=clear_input_field)

    # Sidebar for file upload
    uploaded_audio = st.sidebar.file_uploader('Upload an audio file', type=['wav', 'mp3', 'ogg', 'm4a'])
    uploaded_image = st.sidebar.file_uploader('Upload an image file', type=['jpeg', 'jpg', 'png'])

    # Process uploaded audio
    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        llm_chain.run('Summarize this text: ' + transcribed_audio)

    # Process voice recording
    if voice_recording:
        try:
            transcribed_audio = transcribe_audio(voice_recording['bytes'])
            print(transcribed_audio)
            llm_chain.run(transcribed_audio)
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")

    # Process input if send button or state triggered
    if st.session_state.send_input or send_button:
        if uploaded_image:
            with st.spinner('Processing image...'):
                user_message = 'Describe this image in very detailed'
                if st.session_state.user_question != '':
                    user_message = st.session_state.user_question
                    st.session_state.user_question = ''
                llm_answer = handle_image(uploaded_image.getvalue(), st.session_state.user_question)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer)

        if st.session_state.user_question != '':
            response = llm_chain.run(st.session_state.user_question)
            st.session_state.user_question = ''  # Clear the input field after sending

    # Display chat history
    if chat_history.messages:
        with chat_container:
            st.write('Chat History:')
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    # Save chat history
    save_chat_history()

# Run the main function
if __name__ == '__main__':
    main()

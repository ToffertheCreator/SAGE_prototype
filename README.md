# SAGE (Scholarly Assistant Generative Engine) [prototype ver]
your personal research assistant chatbot

# Video Demo
https://mega.nz/file/5Z5yVIjS#_o-4VBbBcxbp3oS2mnp2T2xJKIKIkUTITAAGZgtfGws

# KEY FEATURES
* general chatbot & pdf chatbot
* can handle image and audio inputs
* research paper web scraping
* data analysis and machine learning support

# Installation
* git clone https://github.com/ToffertheCreator/SAGE_prototype.git
* cd sage-chatbot
* python -m venv chat_venv
* source chat_venv/bin/activate
* On Windows use "chat_venv\Scripts\activate"
* if error: type "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process"
* pip install --upgrade pip
* pip install -r requirements.txt
* You will need a .env file in the root of your project directory containing your API tokens for Hugging Face Hub:
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
* for OPENAI (choose only one):
OPENAI_API_KEY=
* streamlit run nav.py

# NOTE
This is a prototype and may still have some bugs or incomplete features. It's an early version, so some functionality might not work as expected. Feel free to contribute and enhance the program by adding improvements or modifying the code to suit your needs. Your contributions are welcome!

# Models i used
* Mistral: https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF
* Embeddings: https://huggingface.co/spaces/mteb/leaderboard
* Audio handler: https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013
* Image handler: https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file [NOTE: >go to the "Multi-modal Models" section and choose the model you want. >download the model as well as the mmproj file, and put them both inside the "llava" directory in the models folder

you may uae any quantized model here: https://huggingface.co/TheBloke

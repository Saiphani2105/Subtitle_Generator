# AI-Powered Subtitle Generator & Search Engine

## 🚀 Project Overview
This AI-powered application is designed to generate subtitles from audio files and search within those subtitles using AI. Built using **Whisper AI** for transcription and **ChromaDB** for efficient search, this project offers a seamless and intuitive experience for managing subtitles.

### Why This Project?
With the increasing demand for accessibility in media content, generating accurate subtitles is crucial for ensuring inclusivity. Additionally, the ability to search within subtitles enhances content discoverability and provides users with an efficient way to locate relevant information.

## 🌟 Features
- **Audio Transcription:** Generate subtitles from audio files (MP3, WAV, AAC, OGG).
- **Subtitle Search:** Perform fast, accurate searches using **ChromaDB** and **Sentence Transformers**.
- **Semantic Search:** Search for words or sentences with contextual understanding using embeddings.
- **Download Options:** Download subtitles in **SRT** or **TXT** formats.
- **Interactive UI:** User-friendly interface built with **Streamlit**.

## 🛠️ Tech Stack
- **Python**: Primary programming language.
- **Streamlit**: For building the interactive user interface.
- **Whisper AI**: For audio transcription.
- **ChromaDB**: For storing and searching subtitle embeddings.
- **Sentence Transformers**: For semantic search using text embeddings.
- **PyDub**: For audio file manipulation.
- **Pickle**: For serializing subtitle embeddings.

## 🧑‍💻 Installation
1. Clone the repository:
    ```bash
    [git clone](https://github.com/Saiphani2105/Subtitle_Generator.git)
    cd subtitle-generator-ai
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage
1. **Start the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
2. **Generate Subtitles:** Upload your audio file to generate accurate subtitles using Whisper AI.
3. **Search Subtitles:** Use the search tab to find specific words or sentences from subtitles using semantic search.
4. **Download Subtitles:** Download the generated subtitles in SRT or TXT format.

## 📦 File Structure
```
- app.py                  # Main Streamlit application
- requirements.txt        # List of dependencies
- subtitle_embeddings.pkl # Pickle file with subtitle embeddings
- chroma_db/              # ChromaDB storage
```

## 🙏 Acknowledgments
- Thanks to **Innomatics** for providing the platform to build this project.
- Special thanks to **Kanav Bansal** for mentorship and guidance.

## 🚀 Live Demo
Check out the live demo on [Hugging Face](https://huggingface.co/spaces/Phaneendrabayi/Subtitle_Search_Engine).

## 📬 Contact
For any feedback or issues, feel free to reach out via [LinkedIn](www.linkedin.com/in/bai-phaneendra) or raise an issue on GitHub.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Saiphani2105/Subtitle_Generator/blob/main/LICENSE) file for details.

---

Made with ❤️ by Bai Phaneendra


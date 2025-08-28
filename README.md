# nursing-empathy-evaluation

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://your-deployment-link-here)

> **Live App:** [https://your-deployment-link-here](https://your-deployment-link-here)
  
## System Requirement: ffmpeg

This project requires the `ffmpeg` binary to be installed and available in your system PATH for audio processing.

### How to install ffmpeg

**Windows:**
1. Download the latest static build from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
2. Extract the ZIP (e.g., to `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH (Environment Variables)
4. Open a new terminal and run `ffmpeg -version` to verify

**Mac:**
```sh
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```sh
sudo apt-get update && sudo apt-get install ffmpeg
```

If `ffmpeg` is not installed, audio features will not work and you may see errors.
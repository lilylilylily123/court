# **Court**

## **AI-based, unbiased, and transparent video analysis**


## _Installation_

Clone the repository:
```bash
git clone https://github.com/lilylilylily123/court.git
```
> Make sure you have a mp4 file ready for analysis, preferably in the same directory as Court.

## IMPORTANT
### ffmpeg Installation
Ensure that you have ffmpeg installed on your system. You can download it from [here](https://ffmpeg.org/download.html).

### Whisper Installation

Clone the repository:
```bash
git clone https://github.com/ggerganov/whisper.cpp
```
Build the project:
```bash
cd whisper.cpp
bash models/download-ggml-model.sh base
make
```
Your whisper executable path should be:
```bash
./build/bin/whisper-cli
```
but we recommend using absolute paths.

## **.env setup**
### Follow .env.example


# _**Usage**_
```bash
python3 video_analyzer.py your_video.mp4
```
## _**Output**_
### A file named `summary.txt` will contain the summary of the video.
> #### For more details, you can check the transcript.txt file, which contains the raw Whisper transcription.


# _**Warning:**_
## This project is not token-friendly, and can consume quite a lot of tokens, especially when analyzing longer videos. Ensure that you have enough tokens available before analyzing a video.


# _**Credits**_

Whisper: _https://github.com/ggerganov/whisper.cpp_


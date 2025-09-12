# LipGen: Automatic Active Speaker Sentence Video Generator

## Overview

**LipGen** is a pipeline for generating **sentence-level active speaker video clips** from raw videos.  
The system combines **speech transcription** with **active speaker detection** so that each spoken sentence is matched to the **correct face crop video** of the speaker.

This allows us to:
- Automatically extract **sentence-aligned video segments** of the speaker’s face.
- Support **in-the-wild videos** such as TV programs or interviews.
- Use the outputs for **lip-reading datasets**, **communication research**, or other multimodal AI applications.

---

## How It Works

The workflow integrates two main components:

1. **Active Speaker Detection (LR-ASD)**  
   - Based on the [LR-ASD repository](https://github.com/Junhua-Liao/LR-ASD).  
   - Detects faces, tracks them across frames, and estimates which speaker is active.  
   - Produces intermediate files:  
     - `pyavi/` : resampled 25fps video and audio (`audio.wav`).  
     - `pyframes/` : extracted frames.  
     - `pywork/` : metadata (`faces.pckl`, `tracks.pckl`, `scores.pckl`, etc.).  
     - `pycrop/` : cropped face track videos.  

2. **Speech Transcription (Whisper)**  
   - Runs on the **resampled audio** produced by LR-ASD (`pyavi/audio.wav`).  
   - Transcribes full speech into **sentence-level segments** with start and end times.  
   - Output saved in `manifest.json`.

3. **Sentence-to-Speaker Alignment**  
   - Each transcribed sentence is mapped to the **active speaker track** overlapping that time window.  
   - A video clip is generated that contains **only the active speaker’s face** for the duration of the sentence.  
   - Audio is preserved in the output, ensuring the clip matches what the speaker said.

---

## Main Script

Our main working script is:

`bash
easy_sentence_clips_final.py`


### Features
- Input: a raw video file.  
- Output: sentence-level **active speaker clips** (video + audio).  
- Automatically handles LR-ASD processing and Whisper transcription.  
- Supports an optional flag:
  - `--skip_asd`: skips LR-ASD if `pywork/`, `pyavi/`, etc. already exist (saves time).

---

## Workflow Summary

1. **Run LR-ASD** to generate resampled video/audio and face tracks.  
2. **Run Whisper** on resampled audio to get sentence timestamps.  
3. **Match sentences with active speakers** using LR-ASD scores and track metadata.  
4. **Export clips** containing only the speaking face with aligned audio.

---

## Example Usage

```bash
# Standard usage (runs LR-ASD + Whisper + alignment)
python easy_sentence_clips_final.py input_video.mp4

# Skip LR-ASD if preprocessing was already done
python easy_sentence_clips_final.py input_video.mp4 --skip_asd
```

## Outputs
- **manifest.json :**
 Whisper transcription aligned with resampled audio.
- **clips/ :** Contains one video per spoken sentence, showing the active speaker’s cropped face with audio.


## Applications
- Lip-reading dataset generation (Japanese or other languages).
- Assistive technology for speech-impaired communication.
- Media analysis (TV programs, interviews, panel discussions).
- Multimodal AI research combining audio and visual cues.

## Roadmap
- [x] Basic LR-ASD integration
- [x] Whisper transcription with sentence segmentation
- [x] Sentence-to-speaker alignment
- [x] Export video clips with audio
- [ ] Refine alignment in challenging “in-the-wild” videos
- [ ] Merge into a single production-ready `LipGen.py` script


## References
- LR-ASD: Active Speaker Detection
- OpenAI Whisper


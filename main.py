import cv2
from deepface import DeepFace
import pygame
import os
import random
import time

# Initialize pygame mixer
pygame.mixer.init()

# Emotion folders and their songs
emotion_songs = {
    'happy': ['songs/happy/happy.mp3', 'songs/happy/happy2.mp3','songs/happy/happy3.mp3','songs/happy/happy4.mp3'],
    'sad': ['songs/sad/sad1.mp3', 'songs/sad/sad2.mp3','songs/sad/sad3.mp3','songs/sad/sad4.mp3',],
    'angry': ['songs/angry/angry1.mp3', 'songs/angry/angry2.mp3', 'songs/angry/angry3.mp3', 'songs/angry/angry4.mp3'],
    'neutral': ['songs/neutral/neutral.mp3', 'songs/neutral/neutral2.mp3', 'songs/neutral/neutral3.mp3', 'songs/neutral/neutral4.mp3'],
}

# Allowed emotions
allowed_emotions = ["happy", "sad", "angry", "neutral"]

# Play a song for the detected emotion
def play_song(emotion):
    if emotion in emotion_songs:
        song = random.choice(emotion_songs[emotion])
        print(f"🎵 Playing {emotion} song: {song}")
        pygame.mixer.music.load(song)
        pygame.mixer.music.play()
    else:
        print(f"No song for emotion: {emotion}")

# Start webcam
cap = cv2.VideoCapture(0)
print("📷 Starting camera... Please look at the webcam.")

current_emotion = None
last_detected_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame from camera.")
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (300, 300))

    # Detect emotion every 5 seconds
    if time.time() - last_detected_time > 3:
        try:
            result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)
            detected_emotion = result[0]['dominant_emotion']

            # Filter to only allowed emotions
            if detected_emotion not in allowed_emotions:
                detected_emotion = "neutral"

            print(f"🧠 Detected Emotion: {detected_emotion}")
            last_detected_time = time.time()

            # If emotion changed, play new song
            if detected_emotion != current_emotion:
                current_emotion = detected_emotion
                pygame.mixer.music.stop()
                play_song(current_emotion)

        except Exception as e:
            print("Error detecting emotion:", e)

        # Full analysis but only show allowed emotions
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        all_emotions = {k: v for k, v in analysis[0]['emotion'].items() if k in allowed_emotions}
        dominant_emotion = max(all_emotions, key=all_emotions.get)
        print("🧠 Filtered Emotions:", all_emotions)
        print("🧠 Dominant Emotion:", dominant_emotion)

    # Show emotion on webcam window
    if current_emotion:
        cv2.putText(frame, f'Emotion: {current_emotion}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow('Emotion Music Player - Press Q to Quit', frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()

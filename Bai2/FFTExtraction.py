import librosa
import numpy as np
import matplotlib.pyplot as plt

def segment_vowel_silence(audio, Fs, threshold = 0.04, min_duration=0.3):
    
    print(len(audio))

    # Chia khung tín hiệu, mỗi khung độ dài 20ms
    frame_length = int(0.02 * Fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)

    print(len(frames))

    # Tính STE từng khung
    ste = np.sum(np.square(frames), axis=0)

    # Chuẩn hóa STE
    ste_normalized = (ste - np.min(ste)) / (np.max(ste) - np.min(ste))
    print(ste_normalized)

    # Phân loại thành tiếng nói và khoảng lặng
    is_speech = ste_normalized > threshold

    is_speech_full = np.repeat(is_speech, frame_length)[:len(audio)]
    is_speech_full = np.pad(is_speech_full, (0, len(audio) - len(is_speech_full)), constant_values=False) 
    

    # Tìm danh sách khoảng lặng
    silence_segments = librosa.effects.split(audio, top_db=threshold)

    # Bỏ đi các khoảng lặng < 300 ms
    for start, end in silence_segments:
        duration = librosa.samples_to_time(end - start, sr=Fs)
        if duration < min_duration:
            is_speech_full[start:end] = True

    # Trả về tín hiệu chỉ chứa nguyên âm hay tiếng nói
    vowel = audio[is_speech_full]
    return vowel

def FFT_1_vowel_1_speaker(audio, Fs, N_FFT=512):
    """
    Hàm Trích xuất vector FFT của 1 nguyên âm 1 người (1 audio input)
    """

    # Chia khung tín hiệu, mỗi khung độ dài 20ms
    frame_length = int(0.02 * Fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)
    #Số khung
    N = frames.shape[1]
    #Chọn vùng ở giữa, M = N//3 khung
    M = N//3

    # Tính biến đổi Fourier nhanh (FFT) từng khung
    fft_frames = []
    for frame in frames[M:2*M]:
        fft_result = np.fft.fft(frame, N_FFT)
        fft_frames.append(fft_result)

    # Tính trung bình cộng của M vector FFT
    avg_fft = np.mean(fft_frames, axis=0)

    return avg_fft
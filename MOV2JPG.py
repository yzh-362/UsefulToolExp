import cv2
import os
import math

def extract_frames_at_interval(video_path: str, output_directory: str, interval_seconds: float) -> None:
    """
    按照指定的时间间隔从视频中提取帧，并将其保存为图像文件。

    参数:
        video_path (str): 输入视频文件的绝对或相对路径。
        output_directory (str): 存储提取帧的输出目录。
        interval_seconds (float): 提取帧之间的时间间隔（单位：秒）。
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"无法打开或读取视频文件: {video_path}")

    # 获取视频的原始帧率与总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise ValueError("从视频文件中获取的帧率无效。")

    # 根据时间间隔和帧率计算跳跃的帧数步长
    frame_step = math.floor(fps * interval_seconds)

    if frame_step == 0:
        frame_step = 1

    current_frame = 0
    saved_count = 0

    while current_frame < total_frames:
        # 将视频流位置设置为当前需要读取的帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        # 构造输出文件名，使用6位数字进行零填充以保证文件系统排序正确
        output_filename = os.path.join(output_directory, f"frame_{saved_count:06d}.jpg")

        # 将帧数据写入磁盘
        cv2.imwrite(output_filename, frame)

        saved_count += 1
        current_frame += frame_step

    cap.release()
    print(f"帧提取过程完成。共提取并保存了 {saved_count} 帧。")

if __name__ == "__main__":
    # 配置输入路径、输出路径和时间间隔
    INPUT_VIDEO_PATH =
    OUTPUT_DIR = "extracted_frames"
    INTERVAL_IN_SECONDS = 30.0
    
    extract_frames_at_interval(INPUT_VIDEO_PATH, OUTPUT_DIR, INTERVAL_IN_SECONDS)
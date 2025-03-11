import cv2
import numpy as np
import time
import datetime

class AdaptiveFpsCamera:
    def __init__(self):
        # 摄像头参数
        self.cap = cv2.VideoCapture(1)  # 使用默认摄像头
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头！")
        
        # 获取摄像头原始帧率和分辨率
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头原始帧率: {self.original_fps}, 分辨率: {self.width}x{self.height}")
        
        # 背景模型和形态学操作参数
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500,     # 历史帧数
            varThreshold=16, # 方差阈值
            detectShadows=False  # 不检测阴影
        )
        self.kernel = np.ones((5, 5), np.uint8)  # 形态学操作的核
        
        # 帧率控制参数
        self.min_fps = 1        # 最低帧率
        self.max_fps = 10       # 最高帧率
        self.motion_threshold_min = 0.001  # 最小运动占比阈值
        self.motion_threshold_max = 0.05   # 最大运动占比阈值
        self.current_fps = self.min_fps    # 当前帧率
        self.smoothing_factor = 0.9        # 平滑因子，越大变化越慢
        
        # 时间和计数器
        self.last_frame_time = time.time()       # 上一帧的时间
        self.fps_update_interval = 1.0           # 计算实际FPS的时间间隔(秒)
        self.frame_count = 0                     # 用于计算实际FPS的帧计数
        self.last_fps_update_time = time.time()  # 上次更新显示FPS的时间
        self.actual_fps = 0                      # 实际测量的FPS
        
        # 创建窗口
        cv2.namedWindow('Monitor', cv2.WINDOW_NORMAL)
    
    def start(self):
        print("按 'q' 退出程序")
        
        try:
            while True:
                # 计算当前应有的帧间隔
                required_interval = 1.0 / self.current_fps
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                # 如果还没到下一帧的时间，等待
                if elapsed < required_interval:
                    time.sleep(0.001)  # 短暂休眠避免CPU占用过高
                    continue
                
                # 读取当前帧
                self.last_frame_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头画面!")
                    break
                
                # 帧计数增加
                self.frame_count += 1
                
                # 计算实际FPS
                if current_time - self.last_fps_update_time >= self.fps_update_interval:
                    self.actual_fps = self.frame_count / (current_time - self.last_fps_update_time)
                    self.frame_count = 0
                    self.last_fps_update_time = current_time
                
                # 应用背景减除和运动检测（不显示）
                fgmask = self.fgbg.apply(frame)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
                
                # 计算运动量 (前景像素占比)
                motion_ratio = np.count_nonzero(fgmask) / float(fgmask.size)
                
                # 计算新的目标帧率 - 非线性映射
                if motion_ratio < self.motion_threshold_min:
                    target_fps = self.min_fps
                elif motion_ratio > self.motion_threshold_max:
                    target_fps = self.max_fps
                else:
                    # 在最小和最大阈值之间进行线性插值
                    motion_scale = (motion_ratio - self.motion_threshold_min) / (self.motion_threshold_max - self.motion_threshold_min)
                    target_fps = self.min_fps + motion_scale * (self.max_fps - self.min_fps)
                
                # 平滑帧率变化
                self.current_fps = self.smoothing_factor * self.current_fps + (1 - self.smoothing_factor) * target_fps
                
                # 在帧上显示当前帧率（大字体）
                fps_text = f"Current: {self.current_fps:.1f} FPS"
                cv2.putText(frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.2, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 显示原始画面
                cv2.imshow('Monitor', frame)
                
                # 按键处理
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # 释放资源
            self.release()
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    try:
        camera = AdaptiveFpsCamera()
        camera.start()
    except Exception as e:
        print(f"程序出错: {e}")
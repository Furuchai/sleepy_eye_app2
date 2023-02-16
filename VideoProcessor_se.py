import mediapipe as mp
import cv2
import numpy as np
import av
from playsound import playsound # playsound==1.2.2 *1.3.0ではエラー発生

# mediapipe_poseのクラス
class face_mesh_VideoProcessor:
    def __init__(self) -> None:
        self.slpy_frame = int(0)
        self.judge_time = int(20)
        self.judge_eye = 0.3

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # face_mesh のインスタンス
        mp_face_mesh = mp.solutions.face_mesh
        # tick = cv2.getTickCount()
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,) as face_mesh:
            # 左右逆転
            img = cv2.flip(img, 1)
            # 画角の高さ
            height = img.shape[0]
            # 画角の幅
            width = img.shape[1]
            # Face_Meshの座標抽出
            results = face_mesh.process(img)
            
            # 眠気判定関数の定義
            def sleepy_eye(upper, lower, inside, outside):
                np_upper = np.array(upper)
                np_lower = np.array(lower)
                np_inside = np.array(inside)
                np_outside = np.array(outside)
                open_close = np.linalg.norm(np_upper-np_lower) / np.linalg.norm(np_inside-np_outside)
                return open_close

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
            #  右目座標：上386、下374、内362、外263、左目座標：上159、下145、内133、外33
            #  右目判定
                    right_eye_upper = results.multi_face_landmarks[0].landmark[386]
                    right_eye_lower = results.multi_face_landmarks[0].landmark[374]
                    right_eye_in = results.multi_face_landmarks[0].landmark[362]
                    right_eye_out = results.multi_face_landmarks[0].landmark[263]
                    r_upper = [right_eye_upper.x,right_eye_upper.y]
                    r_lower = [right_eye_lower.x, right_eye_lower.y]
                    r_in = [right_eye_in.x, right_eye_in.y]
                    r_out = [right_eye_out.x, right_eye_out.y]
                    r_eye_jd = sleepy_eye(r_upper, r_lower, r_in, r_out)
            # 左目判定
                    left_eye_upper = results.multi_face_landmarks[0].landmark[159]
                    left_eye_lower = results.multi_face_landmarks[0].landmark[145]   
                    left_eye_in = results.multi_face_landmarks[0].landmark[133]
                    left_eye_out = results.multi_face_landmarks[0].landmark[33]   
                    l_upper = [left_eye_upper.x, left_eye_upper.y]
                    l_lower = [left_eye_lower.x, left_eye_lower.y]
                    l_in = [left_eye_in.x, left_eye_in.y]
                    l_out = [left_eye_out.x, left_eye_out.y]
                    l_eye_jd = sleepy_eye(l_upper, l_lower, l_in, l_out) 

                # 判定値の画面表示
                    cv2.putText(img, "left eye:{:.2f} ".format(l_eye_jd), 
                        (10, height-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA) 
                    cv2.putText(img, "right eye:{:.2f} ".format(r_eye_jd), 
                        (int(width/2), height-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA) 

                # FPS計算表示    
                    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
                    # cv2.putText(img, "FPS:{} ".format(int(fps)), 
                    #             (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2, cv2.LINE_AA)   
                    
                    # 目を閉じている状態の判定
                    if (r_eye_jd < self.judge_eye) and (l_eye_jd < self.judge_eye):
                    # self.slpy_frameをカウントアップ。
                        self.slpy_frame += 1
                    # カウントアップ状況の画面表示。
                        cv2.putText(img, "slpy_frame:{:.0f} ".format(self.slpy_frame), 
                            (10, int(height/2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA) 
                        if self.slpy_frame > self.judge_time:
                    #        playsound('bow.mp3')
                            cv2.putText(img,"Wake Up!!", 
                                (int(width/2), int(height/2)), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_AA)                           
                    else:
                        self.slpy_frame = 0
                        continue

        return av.VideoFrame.from_ndarray(img, format="bgr24")

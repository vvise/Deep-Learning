import cv2
import mediapipe as mp

#핸드 이미지 위에 랜드마크 그리기 위함
mp_drawing=mp.solutions.drawing_utils
#핸드 처리
mp_hands= mp.solutions.hands
#핸드 렌드마크 표시 스타일용
drawing_styles=mp.solutions.drawing_styles

#웹캠 열기
cap=cv2.VideoCapture(0)

#손가락 솔루션
with mp_hands.Hands(
    #두손 인식
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    #카메라 열려있을때까지 루프
    while cap.isOpened():
        #카메라에서 사진 한장 얻기
        success, image=cap.read()
        #사진을 얻어오지 못했다면
        if not success:
            #에러 출력 후 루프 시작으로 되돌아감
            #카메라 로딩중일 수 있기 때문에 break 대신 continue
            print("Ignoring empty camera frame.")
            continue
        #사진을 좌/우 반전 시킨 후 BRG에서 RGB로 변환
        image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_RGB2BGR)
        #성능 향상을 위해 이미지를 쓰기 불가시켜 참조로 전달
        image.flags.writeable=False
        #손가락 마디 검출
        result=hands.process(image) 

        #다시 이미지를 쓰기 가능으로 변경
        image.flags.writeable=True
        #이미지를 다시 RGB에서 BGR로 변경
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #hands.process에서 랜드마크를 찾앗다면
        if result.multi_hand_landmarks:
            #랜드마크 숫자 만큼 루프
            for hand_landmarks in result.multi_hand_landmarks:
                #이미지에 랜드마크 위치마다 표시
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style())
        #최종 변경된 이미지를 화면에 출력
        cv2.imshow('Mediapipe Hand', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
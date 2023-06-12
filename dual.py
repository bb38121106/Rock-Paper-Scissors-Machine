import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 6
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        rps_result = []

        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        # Who wins?
        if len(rps_result) >= 2:
            winners = []
            text = ''
            
            s = (3,len(rps_result))
            rps_result_count = np.zeros(s)	
            for i in range(0,len(rps_result)):
                if rps_result[i]['rps']=='rock':
                	rps_result_count[0][i] = 1
                elif rps_result[i]['rps']=='paper':
                	rps_result_count[1][i] = 1
                elif rps_result[i]['rps']=='scissors':
                	rps_result_count[2][i] = 1
            
            rcnt = int(np.sum(rps_result_count[0]))
            pcnt = int(np.sum(rps_result_count[1]))
            scnt = int(np.sum(rps_result_count[2]))
            
            print("-----")
            print("rock", rcnt)
            print("paper", pcnt)
            print("scissors", scnt)
            
            if rcnt > 0 and pcnt > 0 and scnt > 0:
                text = 'Tie'
                print("tie1")
            elif rcnt > 0 and pcnt == 0 and scnt == 0:
                text = 'Tie'
                print("tie2")
            elif rcnt == 0 and pcnt > 0 and scnt == 0:
                text = 'Tie'
                print("tie3")
            elif rcnt == 0 and pcnt == 0 and scnt > 0:
                text = 'Tie'
                print("tie4")
            elif rcnt > 0 and scnt > 0 and pcnt == 0:
                text = 'Rock wins'
                print("Rock wins")
                for i in range(0,len(rps_result)):
                    if rps_result_count[0][i] == 1:
                        winners.append(i)
            elif rcnt > 0 and pcnt > 0 and scnt == 0:
                text = 'Paper wins'
                print("Paper wins")
                for i in range(0,len(rps_result)):
                    if rps_result_count[1][i] == 1:
                        winners.append(i)
            elif pcnt > 0 and scnt > 0 and rcnt == 0:
                text = 'Scissors wins'
                print("Scissors wins")
                for i in range(0,len(rps_result)):
                    if rps_result_count[2][i] == 1:
                        winners.append(i)
                        
            print(winners)
            print("-----")
            
            if winners:
                for winner in winners:
                    cv2.putText(img, text='Winner', org=(rps_result[winner]['org'][0], rps_result[winner]['org'][1] + 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
            cv2.putText(img, text=text, org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)                
            	
    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break

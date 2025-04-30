import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from key import Key
import time

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1580)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 920)

colorKey = (232, 162, 0)
colorHighlight = (0, 255, 0)

start_x, start_y, w, h = 50, 300, 60, 60
textWritten = ""
padding = 80
x_gap = 0

# Create hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# List of key
keyList = []
alphabet = ['A', 'Z', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'Q', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'W', 'X', 'C', 'V', 'B', 'N', 'De']
for key_index, label in enumerate(alphabet):
    if key_index % 10 == 0: x_gap = 0
    keyList.append(
        Key(
            [start_x + x_gap * padding, start_y if key_index < 10 else (start_y + padding) if key_index < 20 else (start_y + padding * 2)],
            [w, h],
            label
        )
    )
    x_gap += 1

def draw_keyboard(img):
    # Draw transparency rectangle
    imgNew = np.zeros_like(img, np.uint8)
    for rect in keyList:
        x, y = rect.posCenter
        w, h = rect.size

        # Choose color based on highlight state
        color = colorHighlight if rect.highlighted else colorKey

        cv2.rectangle(imgNew, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, cv2.FILLED)
        # cvzone.cornerRect(imgNew, (x - w // 2, y - h // 2, w, h), 10, rt=0)

        text = rect.label

        # Choose font and size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3

        # Calculate text size to center it
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2

        # Draw the text on the rectangle
        cv2.putText(imgNew, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    out = img.copy()
    alpha = 0.3
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    return out


def draw_text_display(img, text, opacity=0.7):
    # Draw a white rectangle at the top center to display text
    h, w, _ = img.shape

    display_w = 600
    display_h = 60
    display_x = (w - display_w) // 2
    display_y = 20

    overlay = img.copy()
    cv2.rectangle(overlay, (display_x, display_y), (display_x + display_w, display_y + display_h), (255, 255, 255), cv2.FILLED)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # Draw border
    cv2.rectangle(img, (display_x, display_y), (display_x + display_w, display_y + display_h), (200, 200, 200), 2)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate text position to center it
    text_x = display_x + (display_w - text_size[0]) // 2
    text_y = display_y + (display_h + text_size[1]) // 2

    # Draw the text (black for better visibility on white background)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return img

while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)  # Flip to mirror the image

    # Find the hands
    hands, image = detector.findHands(image)

    # Draw the text display area
    image = draw_text_display(image, textWritten)

    # If hands are detected
    if hands:
        for hand in hands:
            lmList = hand["lmList"] # finger coordinates
            fingerType = hand["type"]
            if lmList:
                if fingerType == "Right":
                    # Extract the x,y coordinates from finger 8 and 12
                    index_finger = (lmList[8][0], lmList[8][1])
                    middle_finger = (lmList[12][0], lmList[12][1])

                    # Find distance between landmark 8 (index finger) and 12 (middle finger)
                    distance, _, _ = detector.findDistance(index_finger, middle_finger, image)

                    if distance < 50:
                        # Check if index finger is over any key
                        for key in keyList:
                            selectedKey = key.check_active(middle_finger)
                            if selectedKey:
                                print(selectedKey.label)
                                text = selectedKey.label
                                if text == 'De':
                                    textWritten = textWritten[:-1]
                                else:
                                    textWritten += text
                                # Avoid many text writing
                                time.sleep(0.05)
                else:
                    index_finger = (lmList[8][0], lmList[8][1])
                    inch_finger = (lmList[4][0], lmList[4][1])

                    distance, _, _ = detector.findDistance(index_finger, inch_finger, image)

                    if distance < 60:
                        # Update the selected rectangle
                        for rect in keyList:
                            rect.update(inch_finger)


    out = draw_keyboard(image)

    cv2.imshow("Image", out)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

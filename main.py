import cv2
from cvzone.HandTrackingModule import HandDetector

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# Outer var
colorR = (33, 33, 33)
colorKey = (232, 162, 0)
colorHighlight = (0, 255, 0)
textWritten = ""

# Create hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

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

def draw_key(img, text, x, y, highlighted=False, opacity=0.5):
    key_w = 45
    key_h = 45

    # Choose color based on highlight state
    color = colorHighlight if highlighted else colorKey

    # Create a transparent layer
    overlay = img.copy()

    # Draw filled rectangle on the overlay
    cv2.rectangle(overlay, (x, y), (x + key_w, y + key_h), color, cv2.FILLED)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # Draw border (fully opaque)
    cv2.rectangle(img, (x, y), (x + key_w, y + key_h), colorKey, 2)

    # Add text in the center
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # Get text size to calculate center position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x + (key_w - text_size[0]) // 2
    text_y = y + (key_h + text_size[1]) // 2

    # Draw text outline for better visibility
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)

    # Draw the text
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return x, y, x + key_w, y + key_h


def draw_keyboard(img, keys):
    # Draw all keys with their current highlight state
    key_bounds = []
    for key_map in keys:
        bounds = draw_key(img, key_map["text"], key_map["x"], key_map["y"], key_map["highlighted"])
        key_bounds.append(bounds)

    return key_bounds

def check_key_hover(finger_pos, key_bounds):
    # Check if finger is hovering over any key
    x, y = finger_pos
    for i, (x1, y1, x2, y2) in enumerate(key_bounds):
        if x1 < x < x2 and y1 < y < y2:
            return i
    return -1

# Store key positions and states
alphabet = ['A', 'Z', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'Q', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'W', 'X', 'C', 'V', 'B', 'N', '<']
keys = []
start_x = 50
start_y = 300
padding = 50
x_gap = 0

for key_index, letter in enumerate(alphabet):
    if key_index % 10 == 0: x_gap = 0
    keys.append({
        "text": letter,
        "x":  start_x + x_gap * padding,
        "y": start_y if key_index < 10 else (start_y + padding) if key_index < 20 else (start_y + padding * 2),
        "highlighted": False
    })
    x_gap += 1


while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)  # Flip to mirror the image

    # Find the hands
    hands, image = detector.findHands(image)

    # Reset all keys to not highlighted
    for key in keys:
        key["highlighted"] = False

    # Draw and get key bounds
    key_bounds = draw_keyboard(image, keys)

    # Draw the text display area
    image = draw_text_display(image, textWritten)

    # If hands are detected
    if hands:
        for hand in hands:
            lmList = hand["lmList"] # finger coordinates
            if lmList:
                # Extract the x,y coordinates from finger 8 and 12
                index_finger = (lmList[8][0], lmList[8][1])
                middle_finger = (lmList[12][0], lmList[12][1])

                # Find distance between landmark 8 (index finger) and 12 (middle finger)
                distance, _, _ = detector.findDistance(index_finger, middle_finger, image)

                if distance < 50:
                    # Check if index finger is over any key
                    key_index = check_key_hover(index_finger, key_bounds)
                    if key_index >= 0:
                        keys[key_index]["highlighted"] = True
                        text = keys[key_index]['text']
                        if text == '<':
                            textWritten = textWritten[:-1]
                        else:
                            textWritten += keys[key_index]['text']
                            print(f"Key {keys[key_index]['text']} clicked")

                # Redraw the keyboard with updated highlights
                draw_keyboard(image, keys)

    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

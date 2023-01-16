from taipy import Gui
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

selected_file = None
content = None


def process_image(state):
    img = cv2.imread(state.selected_file, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 10)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]

        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # To draw a rectangle in eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 10)

    state.content = cv2.imencode(".jpg", img)[1].tobytes()


content = """
<|{selected_file}|file_selector|label=Upload File|on_action=process_image|extensions=.jpg,.gif,.png|drop_message=Drop Message|>
<|{content}|image|width=300px|height=300px|>
"""


if __name__ == "__main__":
    Gui(page=content).run(dark_mode=False, port=8080)

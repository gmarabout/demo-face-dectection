from taipy import Gui
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

number_of_faces_detected = 0
selected_file = None
img = None


def process_image(state):
    img = cv2.imread(state.selected_file, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    state.number_of_faces_detected = len(faces)
    # Draw a rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 10)

    state.img = cv2.imencode(".jpg", img)[1].tobytes()


content = """
<|{selected_file}|file_selector|label=Upload File|on_action=process_image|extensions=.jpg,.gif,.png|drop_message=Drop Message|>

<|{img}|image|width=300px|height=300px|>

<|{number_of_faces_detected} face(s) detected|>
"""


if __name__ == "__main__":
    Gui(page=content).run(dark_mode=False, port=8080)

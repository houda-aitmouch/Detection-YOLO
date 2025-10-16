from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialisation de la caméra
cap = cv2.VideoCapture(0)  # Essaye avec 0 pour une webcam par défaut ou 1, 2 pour d'autres caméras
cap.set(3, 1280)
cap.set(4, 720)

# Charger le modèle YOLOv8
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Noms des classes à détecter
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

# Vérifier si la caméra est ouverte
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra")
    exit()

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:
        print("Erreur lors de la lecture de l'image de la caméra.")
        break

    # Obtenir les résultats de la détection
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordonnées de la boîte de détection
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Dimensions de la boîte
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confiance du modèle
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Classe de l'objet détecté
            cls = int(box.cls[0])

            # Ajouter le texte avec le nom de la classe et la confiance
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculer les FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Affichage des FPS sur l'image
    cvzone.putTextRect(img, f"FPS: {int(fps)}", (10, 40), scale=1, thickness=1)

    # Affichage de l'image avec les résultats de la détection
    cv2.imshow("Image", img)

    # Sortir de la boucle avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO v8
model = YOLO("yolov8n.pt")

def detectar_queda(caixa):
    x1, y1, x2, y2 = caixa
    largura = x2 - x1
    altura = y2 - y1
    # Verifica se a altura é menor que a largura para identificar uma possível queda
    if altura < largura:
        return True
    return False

# Carregar o vídeo
cap = cv2.VideoCapture("./video/teste.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Configurar o gravador de vídeo
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))

# Processar cada quadro do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção com YOLO
    resultados = model(frame)

    # Processar as detecções
    for caixa, class_idx in zip(resultados[0].boxes.xyxy, resultados[0].boxes.cls):
        x1, y1, x2, y2 = map(int, caixa)
        class_idx = int(class_idx)

        # Verificar se a detecção é de uma pessoa (class_idx == 0 para pessoas)
        if class_idx == 0:
            if detectar_queda((x1, y1, x2, y2)):  # Detectar queda
                cv2.putText(frame, "Queda detectada", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                pass

    # Exibir o quadro e salvar no arquivo de saída
    cv2.imshow("Analise de Queda", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

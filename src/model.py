from ultralytics import YOLO
import numpy
import cv2
import torch
import os
import psutil


colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 128, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]



def learning_neuro():
    print("Проверка CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Название GPU:", torch.cuda.get_device_name(0))
    
    # Явно указываем индекс GPU
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = YOLO('./runs/restudying_neuro_v6.6s(1)/weights/best.pt')
    model.train(
        data='data.yaml',
        epochs=7,
        imgsz=1504,
        name='restudying_neuro_v6.7s(1)',
        patience=7,
        batch=10,  # Уменьшенный размер батча
        device='cuda',  # Теперь передаётся как 0 или 'cpu'
        project='./runs',
        workers=6
        #amp=False  # Временно отключено для теста
    )


def process_image(path, test_image):
    try:
        # Предобученная модель
        model = YOLO('./runs/restudying_neuro_v6.7s(1)/weights/best.pt')
        
        # Формирование полного пути к изображению
        full_image_path = os.path.join(path, test_image)
        
        # Проверка существования файла
        if not os.path.exists(full_image_path):
            print(f"❌ Ошибка: Файл не найден: {full_image_path}")
            print(f"   Проверьте путь: {path}")
            print(f"   Имя файла: {test_image}")
            return
        
        # Загрузка изображения
        image = cv2.imread(full_image_path)
        
        # Проверка загрузки изображения
        if image is None:
            print(f"❌ Ошибка: Не удалось загрузить изображение: {full_image_path}")
            print(f"   Возможные причины:")
            print(f"   1. Файл поврежден")
            print(f"   2. Неподдерживаемый формат")
            print(f"   3. Проблемы с правами доступа")
            return
        
        print(f"✅ Изображение успешно загружено: {full_image_path}")
        original_height, original_width = image.shape[:2]  # Сохраняем исходный размер изображения
        print(f"   Размеры: {original_width}x{original_height}")
        
        thickness = 1
        font_scale = 0.5
        IOU_THRESHOLD = 0.5
        SCORE_THRESHOLD = 0.5
        
        # Применение модели
        results = model(image, conf=0.01, classes=[0, 1, 2, 4, 8])[0]
        
        # Получение оригинального изображения и результатов
        image = results.orig_img
        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(numpy.int32)
        
        # Масштабирование bounding boxes к исходному размеру изображения
        scale_x = original_width / results.orig_shape[1]
        scale_y = original_height / results.orig_shape[0]
        boxes = boxes * numpy.array([scale_x, scale_y, scale_x, scale_y])
        boxes = boxes.astype(numpy.int32)
        
        # Словарь для группировки результатов
        grouped_objects = {}
        
        # Группировка результатов по классам
        for class_id, box in zip(classes, boxes):
            class_name = classes_names[int(class_id)]
            if class_name not in grouped_objects:
                grouped_objects[class_name] = []
            grouped_objects[class_name].append(box.tolist())  # Конвертируем numpy array в list
        
        # Подготовка данных для NMS
        boxes_nms = []
        confidences_nms = []
        class_ids_nms = []
        
        # Сбор боксов и уверенности для NMS
        result = results  # У вас уже есть results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Применяем масштабирование и для NMS боксов
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            boxes_nms.append([x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled])
            confidences_nms.append(conf)
            class_ids_nms.append(cls)
        
        # Фильтрация NMS
        idxs = cv2.dnn.NMSBoxes(boxes_nms, confidences_nms, SCORE_THRESHOLD, IOU_THRESHOLD)
        
        # Рисование отфильтрованных боксов
        if idxs is not None and len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes_nms[i]
                cls = class_ids_nms[i]
                conf = confidences_nms[i]
                
                # Выбор цвета для класса
                color = colors[cls % len(colors)]
                color = [int(c) for c in color]
                
                # Нарисовать bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
                
                # Подготовка текста
                text = f"{classes_names[cls]} {conf:.2f}"
                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                           fontScale=font_scale, thickness=thickness)[0]
                
        
                text_offset_x, text_offset_y = x, y - 5
                box_coords = ((text_offset_x, text_offset_y - text_height - 2), 
                             (text_offset_x + text_width + 2, text_offset_y))
                
                # Добавление полупрозрачного фона
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                
                # Отображение текста
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           fontScale=font_scale, color=(255, 255, 255), thickness=thickness)
        
        # Создание директории для сохранения результатов, если её нет
        output_dir = "./yolo_image+text/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохранение измененного изображения
        base_name = os.path.splitext(test_image)[0]
        ext = os.path.splitext(test_image)[1]
        new_image_path = os.path.join(output_dir, f"{base_name}_yolo{ext}")
        cv2.imwrite(new_image_path, image)
        
        # Сохранение данных в текстовый файл
        text_file_path = os.path.join(output_dir, f"{base_name}_yolo_data.txt")
        with open(text_file_path, 'w', encoding='utf-8') as f:
            for class_name, boxes_list in grouped_objects.items():
                f.write(f"{class_name}: {len(boxes_list)} объектов\n")
                for i, box in enumerate(boxes_list, 1):
                    f.write(f"  Объект {i}: x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}\n")
                f.write("\n")
        
        print(f"✅ Обработано: {test_image}")
        print(f"   Изображение сохранено: {new_image_path}")
        print(f"   Данные сохранены: {text_file_path}")
        print(f"   Найдено объектов: {sum(len(v) for v in grouped_objects.values())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обработке {test_image}:")
        print(f"   Тип ошибки: {type(e).__name__}")
        print(f"   Сообщение: {str(e)}")
        import traceback
        traceback.print_exc()
        return False



if __name__ == "__main__":
    learning_neuro()
    
    # folder_path = "./tests"
    # img_list = []

    # for images in os.listdir(folder_path):
    #     if(images.endswith('.png' or '.jpg' or '.jpeg')):
    #         img_list.append(images)
    # folder_path += '/'
    # print(img_list)
    # for i in range(0, len(img_list)):
    #     process_image(folder_path, img_list[i])

    # print("Физические ядра:", psutil.cpu_count(logical=False))
    # print("Логические ядра:", psutil.cpu_count(logical=True))

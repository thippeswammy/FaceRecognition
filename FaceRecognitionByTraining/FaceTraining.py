from ultralytics import YOLO


def Train(data_path="", test_name="test1"):
    # Initialize YOLO model
    model = YOLO("../yolov8n-cls.pt")

    # Train the model with built-in augmentations
    model.train(
        data=data_path,
        epochs=100,
        imgsz=512,
        batch=16,
        workers=48,
        name=test_name,
        augment=True,  # Enable YOLO's default augmentations
        hsv_h=0.015,  # Adjust hue augmentation
        hsv_s=0.7,  # Adjust saturation augmentation
        hsv_v=0.4,  # Adjust value augmentation
        degrees=15,  # Rotate images up to 10 degrees
        translate=0.1,  # Translate images up to 10%
        scale=0.4,  # Scale images by up to 50%
        shear=2,  # Shear images by up to 2 degrees
        flipud=0.0,  # Flip images upside down (set to 0 to disable)
        fliplr=0.5,  # Flip images horizontally with 50% probability
        mosaic=1.0,  # Use mosaic augmentation with 100% probability
        mixup=0.1  # Use mix up augmentation with 20% probability
    )

# Example usage
# Train(data_path="path/to/dataset", test_name="face_classification")

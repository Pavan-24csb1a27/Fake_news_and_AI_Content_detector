
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


class FakeImageDetector(nn.Module):
    def __init__(self):
        super(FakeImageDetector, self).__init__()
        
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.model(x)



def predict_image(image_path, model_path="models/image_model/fake_image_det.pth"):
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Define the exact same transformations used in training
    # (Excluding data augmentation like RandomHorizontalFlip)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load and Initialize Model
    model = FakeImageDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode

    # 4. Load and Preprocess Image
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)
        image = image.to(device)
    except Exception as e:
        return f"Error loading image: {e}"

    # 5. Perform Inference
    with torch.no_grad():
        outputs = model(image)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # 6. Map to Class Names
    # Note: ImageFolder maps folders alphabetically. 
    # Usually: 0 = 'Fake' (or first folder), 1 = 'Real' (or second folder)
    # Check your train directory structure to confirm!
    classes = ['Fake', 'Real'] 
    
    result = {
        "prediction": classes[predicted.item()],
        "confidence": f"{confidence.item() * 100:.2f}%"
    }
    
    return result

# Example usage:
print(predict_image("test_images\\Screenshot 2025-11-29 184149.png"))
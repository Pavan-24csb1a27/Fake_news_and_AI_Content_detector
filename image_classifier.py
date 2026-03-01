import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Data Transformations (Optimized for ResNet/224px)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = 'data' 
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Initialize Model, Scaler, and Optimizer
    model = FakeImageDetector().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Improves generalization
    
    # Use a lower LR (0.0001) for Transfer Learning
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    num_epochs = 10
    print("Starting Transfer Learning training...")
    
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f" > Train Loss: {avg_loss:.4f}")
        print(f" > Val Accuracy: {val_acc:.2f}%")
        print(f" > LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 20)
        
        scheduler.step(avg_loss)

    # 4. Save the Model
    os.makedirs("models/image_model", exist_ok=True)
    torch.save(model.state_dict(), "models/image_model/fake_image_det.pth")
    print("Model saved successfully!")

if __name__ == '__main__':
    train_model()
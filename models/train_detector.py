import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from detect_tumor import TumorDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:",device)

# data augmentation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.ImageFolder(
    "dataset_detection",
    transform=transform
)

train_loader = DataLoader(dataset,batch_size=16,shuffle=True)

model = TumorDetector().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003
)

epochs = 10

for epoch in range(epochs):

    total_loss = 0
    correct = 0
    total = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = loss_fn(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _,pred = torch.max(outputs,1)

        correct += (pred==labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1} Loss {total_loss:.3f} Accuracy {accuracy:.2f}%")

torch.save(model.state_dict(),"models/tumor_detector.pth")

print("Model saved successfully")
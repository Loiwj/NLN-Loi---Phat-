import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Cấu hình thiết bị và đường dẫn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir_1 = '/content/NLN-Loi-Phat/Dataset_1/'
image_dir_2 = '/content/NLN-Loi-Phat/Dataset_2/'
model_path = "/content/EfficientNet_B5_MobileNetV3_ResNet50.pth"
name_log = 'evaluation_log'

# Transform cho ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
        # Load và cấu hình EfficientNet
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
        num_ftrs_efficient = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_ftrs_efficient, 512)
        
        # Load và cấu hình MobileNetV3
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        num_ftrs_mobilenet = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(num_ftrs_mobilenet, 512)
        
        # Load và cấu hình ResNet50
        self.resnet50 = models.resnet50(pretrained=True)
        num_ftrs_resnet = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs_resnet, 512)
        
        # Các lớp fully connected kết hợp
        self.fc1 = nn.Linear(512 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out1 = self.efficientnet(x)
        out2 = self.mobilenet(x)
        out3 = self.resnet50(x)
        combined_out = torch.cat((out1, out2, out3), 1)
        combined_out = torch.relu(self.bn1(self.fc1(combined_out)))
        combined_out = torch.relu(self.bn2(self.fc2(combined_out)))
        combined_out = torch.relu(self.bn3(self.fc3(combined_out)))
        combined_out = torch.relu(self.bn4(self.fc4(combined_out)))
        return self.fc5(combined_out)

def create_dataloader(image_dir, transform):
    """Tạo DataLoader từ đường dẫn thư mục ảnh"""
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4), dataset

def evaluate_model(model, dataloader, dataset_name=""):
    """Đánh giá model và tính các metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Tính toán các metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    
    # In và lưu kết quả
    results = f'Evaluation on {dataset_name}:\nAccuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f} AUC: {auc:.4f}\n'
    print(results)
    with open(f"{name_log}.csv", 'a') as log_file:
        log_file.write(results + '\n')
    
    return all_preds, all_labels

def plot_confusion_matrix(true_labels, pred_labels, classes, dataset_name):
    """Vẽ confusion matrix và lưu thành file"""
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Normalized Confusion Matrix - {dataset_name}')
    plt.savefig(f'{name_log}_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def main():
    # Load datasets
    dataloader_1, dataset_1 = create_dataloader(image_dir_1, transform)
    dataloader_2, dataset_2 = create_dataloader(image_dir_2, transform)
    
    # Khởi tạo model và load weights
    model = CombinedModel(num_classes=len(dataset_1.classes))
    
    # Load state dict
    state_dict = torch.load(model_path)
    # Nếu model được train với DataParallel, cần xử lý key names
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Đánh giá trên cả hai dataset
    print("\nEvaluating on Dataset 1:")
    preds_1, labels_1 = evaluate_model(model, dataloader_1, "Dataset 1")
    plot_confusion_matrix(labels_1, preds_1, dataset_1.classes, "Dataset1")
    
    print("\nEvaluating on Dataset 2:")
    preds_2, labels_2 = evaluate_model(model, dataloader_2, "Dataset 2")
    plot_confusion_matrix(labels_2, preds_2, dataset_2.classes, "Dataset2")

if __name__ == "__main__":
    main()
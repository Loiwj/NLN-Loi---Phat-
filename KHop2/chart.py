import pandas as pd

# Data for Dataset 1
data1 = {
    'Model': ['EfficientnetB4_MobilenetV3', 'EfficientnetB4_MobilenetV2', 'EfficientnetB4_Resnet50', 
              'EfficientnetB5_MobilenetV2', 'EfficientnetB5_MobilenetV3', 'EfficientnetB5_Resnet50'],
    'AUC': [0.9736, 0.9843, 0.9697, 0.9806, 0.9775, 0.9618],
    'F1 Score': [0.9495, 0.9704, 0.9421, 0.9632, 0.9571, 0.9280],
    'Recall': [0.9498, 0.9705, 0.9424, 0.9631, 0.9572, 0.9276],
    'Accuracy': [0.9498, 0.9705, 0.9424, 0.9631, 0.9572, 0.9276],

}

# Data for Dataset 2
data2 = {
    'Model': ['EfficientnetB4_MobilenetV3', 'EfficientnetB4_MobilenetV2', 'EfficientnetB4_Resnet50', 
              'EfficientnetB5_MobilenetV2', 'EfficientnetB5_MobilenetV3', 'EfficientnetB5_Resnet50'],
    'AUC': [0.9158, 0.9658, 0.9092, 0.9408, 0.9066, 0.9303],
    'F1 Score': [0.8346, 0.9316, 0.8192, 0.8838, 0.8118, 0.8644],
    'Recall': [0.8400, 0.9350, 0.8275, 0.8875, 0.8225, 0.8675],
    'Accuracy': [0.8400, 0.9350, 0.8275, 0.8875, 0.8225, 0.8675]
    
}

# Create DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Display DataFrames
print("Dataset 1 Evaluation:")
print(df1)
print("\nDataset 2 Evaluation:")
print(df2)
import matplotlib.pyplot as plt

# Plot Dataset 1
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df1.values, colLabels=df1.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Dataset 1 Evaluation')
plt.savefig('dataset1_evaluation.png')

# Plot Dataset 2
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df2.values, colLabels=df2.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Dataset 2 Evaluation')
plt.savefig('dataset2_evaluation.png')
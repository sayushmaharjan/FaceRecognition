import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import warnings
import random

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = './cropped_faces'
WEIGHTS_DIR = './model_weights_2'
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# Create weights directory
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Custom Dataset
class GeorgiaTechDataset(Dataset):
    def __init__(self, root_dir, subjects, image_indices, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        all_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
        for subject in subjects:
            for idx in image_indices:
                img_name = f's{subject:02d}_{idx:02d}.jpg'
                img_path = os.path.join(root_dir, img_name)
                if img_name in all_files and os.path.exists(img_path):
                    self.images.append(img_path)
                    self.labels.append(subject - 1)
                else:
                    print(f'Warning: Image {img_path} not found')
        
        print(f'Total images loaded: {len(self.images)}')
        if len(self.images) == 0:
            raise ValueError("No images loaded. Check dataset path and file naming.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
       
    
        if self.transform:
            image_transformed = self.transform(image)
        else:
            image_transformed = image
        return image_transformed, label, image

# Collate function
def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return None, None
    # Separate tensors/labels and PIL images
    images_transformed = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    images_pil = [item[2] for item in batch]
    # Collate tensors and labels
    collated = torch.utils.data.dataloader.default_collate(list(zip(images_transformed, labels)))
    return collated, images_pil

# Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load Pre-trained ResNet
def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Extract Features
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch, _ in dataloader:
            if batch is None:
                continue
            inputs, lbls = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Cosine Similarity Scores
def compute_cosine_scores(enrollment_features, verification_features, enrollment_labels, verification_labels):
    genuine_scores = []
    impostor_scores = []
    enrollment_features = enrollment_features / np.linalg.norm(enrollment_features, axis=1, keepdims=True)
    verification_features = verification_features / np.linalg.norm(verification_features, axis=1, keepdims=True)
    for i, (enroll_feat, enroll_label) in enumerate(zip(enrollment_features, enrollment_labels)):
        for j, (verify_feat, verify_label) in enumerate(zip(verification_features, verification_labels)):
            score = 1 - cosine(enroll_feat, verify_feat)
            enroll_label_scalar = int(enroll_label) if isinstance(enroll_label, np.ndarray) else enroll_label
            verify_label_scalar = int(verify_label) if isinstance(verify_label, np.ndarray) else verify_label
            if enroll_label_scalar == verify_label_scalar:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)
    return np.array(genuine_scores), np.array(impostor_scores)

# Plot Score Distributions and ROC
def plot_analysis(genuine_scores, impostor_scores, title_prefix):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(genuine_scores, color='green', label='Genuine', kde=True, stat='density')
    sns.histplot(impostor_scores, color='red', label='Impostor', kde=True, stat='density')
    plt.xlim(0.4, 1.0)
    plt.title(f'{title_prefix} Score Distribution')
    plt.xlabel('Cosine Similarity')
    plt.legend()
    
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{title_prefix} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return roc_auc, thresholds

# Calculate d-prime
def calculate_d_prime(genuine_scores, impostor_scores):
    mean_gen = np.mean(genuine_scores)
    mean_imp = np.mean(impostor_scores)
    std_gen = np.std(genuine_scores)
    std_imp = np.std(impostor_scores)
    d_prime = abs(mean_gen - mean_imp) / ((std_gen + std_imp) / 2)
    return d_prime

# Rank Identification Rates
def compute_rank_rates(enrollment_features, gallery_features, enrollment_labels, gallery_labels, threshold):
    rank1_success = 0
    rank3_success = 0
    total = len(enrollment_features)
    
    for i, (enroll_feat, enroll_label) in enumerate(zip(enrollment_features, enrollment_labels)):
        scores = []
        for j, (gallery_feat, gallery_label) in enumerate(zip(gallery_features, gallery_labels)):
            score = 1 - cosine(enroll_feat, gallery_feat)
            scores.append((score, gallery_label))
        scores.sort(reverse=True)
        
        if scores[0][0] > threshold and scores[0][1] == enroll_label:
            rank1_success += 1
        
        top3 = [s for s in scores[:3] if s[0] > threshold]
        if any(s[1] == enroll_label for s in top3):
            rank3_success += 1
    
    return rank1_success / total, rank3_success / total

# Display Sample Validation Images
def display_validation_images(model, dataloader, device, num_images=6):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 8))
    
    with torch.no_grad():
        for batch, images_pil in dataloader:
            if batch is None:
                continue
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    break
                img = images_pil[i]
                pred_label = preds[i].item()
                true_label = labels[i].item()
                
                plt.subplot(2, 3, images_shown + 1)
                plt.imshow(img)
                plt.title(f'Pred: s{pred_label + 1:02d}\nTrue: s{true_label + 1:02d}')
                plt.axis('off')
                images_shown += 1
            
            if images_shown >= num_images:
                break
    
    plt.tight_layout()
    plt.show()

# Train a single model
def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch, _ in train_loader:
            if batch is None:
                continue
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Main 
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 50

    
    # Subject-Dependent Protocol
    print("Subject-Dependent Protocol")
    train_subjects = list(range(1, 51))
    train_indices = list(range(1, 11))
    test_indices = list(range(11, 16))
    
    train_dataset = GeorgiaTechDataset(DATA_DIR, train_subjects, train_indices, data_transforms['train'])
    test_dataset = GeorgiaTechDataset(DATA_DIR, train_subjects, test_indices, data_transforms['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Train Model
    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
        {'params': model.layer3.parameters(), 'lr': LEARNING_RATE / 10},
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATE / 10}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    train_model(model, train_loader, criterion, optimizer, scheduler, device, NUM_EPOCHS)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'subject_dependent.pth'))
    print(f"Saved subject-dependent model weights to {os.path.join(WEIGHTS_DIR, 'subject_dependent.pth')}")
    
    # Display Sample Validation Images
    print("\nDisplaying Sample Validation Images")
    display_validation_images(model, test_loader, device, num_images=6)
    
    # Extract Features
    model.fc = nn.Identity()
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # Enrollment and Verification
    enrollment_features = []
    enrollment_labels = []
    verification_features = []
    verification_labels = []
    
    for subject in range(50):
        subj_idx = np.where(test_labels == subject)[0]
        if len(subj_idx) >= 5:
            enrollment_features.append(test_features[subj_idx[0]])
            enrollment_labels.append(test_labels[subj_idx[0]])
            verification_features.extend(test_features[subj_idx[1:]])
            verification_labels.extend(test_labels[subj_idx[1:]])
    
    enrollment_features = np.array(enrollment_features)
    enrollment_labels = np.array(enrollment_labels)
    verification_features = np.array(verification_features)
    verification_labels = np.array(verification_labels)
    
    # Cosine Similarity Scores
    genuine_scores, impostor_scores = compute_cosine_scores(
        enrollment_features, verification_features, enrollment_labels, verification_labels)
    
    # Print Score Ranges
    print(f'Genuine scores: min={np.min(genuine_scores):.4f}, max={np.max(genuine_scores):.4f}, mean={np.mean(genuine_scores):.4f}')
    print(f'Impostor scores: min={np.min(impostor_scores):.4f}, max={np.max(impostor_scores):.4f}, mean={np.mean(impostor_scores):.4f}')
    
    # Plot and Compute Metrics
    roc_auc, thresholds = plot_analysis(genuine_scores, impostor_scores, "Subject-Dependent")
    d_prime = calculate_d_prime(genuine_scores, impostor_scores)
    print(f'Subject-Dependent: AUC = {roc_auc:.4f}, d-prime = {d_prime:.4f}')
    
    #  EER Threshold
    fpr, tpr, thresholds = roc_curve(
        np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]),
        np.concatenate([genuine_scores, impostor_scores]))
    eer_threshold = thresholds[np.argmin(np.abs(tpr - (1 - fpr)))]
    print(f'EER threshold: {eer_threshold:.4f}')
    
    # Rank Identification Rates
    gallery_features = verification_features
    gallery_labels = verification_labels
    rank1, rank3 = compute_rank_rates(
        enrollment_features, gallery_features, enrollment_labels, gallery_labels, eer_threshold)
    print(f'Subject-Dependent: Rank-1 = {rank1:.4f}, Rank-3 = {rank3:.4f}')
    
    # PCA Dimensionality Reduction
    print("\nBonus 1: PCA Dimensionality Reduction")
    pca = PCA(n_components=50)
    train_features_pca = pca.fit_transform(train_features)
    test_features_pca = pca.transform(test_features)
    
    enrollment_features_pca = []
    enrollment_labels_pca = []
    verification_features_pca = []
    verification_labels_pca = []
    
    for subject in range(50):
        subj_idx = np.where(test_labels == subject)[0]
        if len(subj_idx) >= 5:
            enrollment_features_pca.append(test_features_pca[subj_idx[0]])
            enrollment_labels_pca.append(int(test_labels[subj_idx[0]]))
            verification_features_pca.extend(test_features_pca[subj_idx[1:]])
            verification_labels_pca.extend([int(label) for label in test_labels[subj_idx[1:]]])
    
    enrollment_features_pca = np.array(enrollment_features_pca)
    enrollment_labels_pca = enrollment_labels_pca
    verification_features_pca = np.array(verification_features_pca)
    verification_labels_pca = verification_labels_pca
    
    genuine_scores_pca, impostor_scores_pca = compute_cosine_scores(
        enrollment_features_pca, verification_features_pca, enrollment_labels_pca, verification_labels_pca)
    
    roc_auc_pca, _ = plot_analysis(genuine_scores_pca, impostor_scores_pca, "Subject-Dependent PCA")
    d_prime_pca = calculate_d_prime(genuine_scores_pca, impostor_scores_pca)
    print(f'Subject-Dependent PCA: AUC = {roc_auc_pca:.4f}, d-prime = {d_prime_pca:.4f}')
    
    # Committee of Models
    print("\nBonus 2: Committee of Models")
    seeds = [49, 123, 456]
    all_genuine_scores = []
    all_impostor_scores = []
    committee_features = []
    committee_labels = []
    
    for seed in seeds:
        print(f"\nTraining model with seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        model = get_model(num_classes).to(device)
        optimizer = optim.Adam([
            {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
            {'params': model.layer3.parameters(), 'lr': LEARNING_RATE / 10},
            {'params': model.layer4.parameters(), 'lr': LEARNING_RATE / 10}
        ])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        train_model(model, train_loader, criterion, optimizer, scheduler, device, NUM_EPOCHS)
        
        # Save model weights
        weight_path = os.path.join(WEIGHTS_DIR, f'committee_model_{seed}.pth')
        torch.save(model.state_dict(), weight_path)
        print(f"Saved committee model weights to {weight_path}")
        
        # Display Sample Validation Images
        print(f"\nDisplaying Sample Validation Images for Model with Seed {seed}")
        display_validation_images(model, test_loader, device, num_images=6)
        
        model.fc = nn.Identity()
        test_features, test_labels = extract_features(model, test_loader, device)
        
        enrollment_features = []
        verification_features = []
        for subject in range(50):
            subj_idx = np.where(test_labels == subject)[0]
            if len(subj_idx) >= 5:
                enrollment_features.append(test_features[subj_idx[0]])
                verification_features.extend(test_features[subj_idx[1:]])
        
        enrollment_features = np.array(enrollment_features)
        verification_features = np.array(verification_features)
        
        genuine_scores, impostor_scores = compute_cosine_scores(
            enrollment_features, verification_features, enrollment_labels, verification_labels)
        
        all_genuine_scores.append(genuine_scores)
        all_impostor_scores.append(impostor_scores)
        
        committee_features.append((enrollment_features, verification_features))
        committee_labels.append((enrollment_labels, verification_labels))
    
    # Average scores 
    avg_genuine_scores = np.mean(all_genuine_scores, axis=0)
    avg_impostor_scores = np.mean(all_impostor_scores, axis=0)
    
    # Average features for Rank Rates
    avg_enrollment_features = np.mean([f[0] for f in committee_features], axis=0)
    avg_verification_features = np.mean([f[1] for f in committee_features], axis=0)
    avg_enrollment_labels = committee_labels[0][0]
    avg_verification_labels = committee_labels[0][1]
    
    # Print Score Ranges
    print(f'Committee Genuine scores: min={np.min(avg_genuine_scores):.4f}, max={np.max(avg_genuine_scores):.4f}, mean={np.mean(avg_genuine_scores):.4f}')
    print(f'Committee Impostor scores: min={np.min(avg_impostor_scores):.4f}, max={np.max(avg_impostor_scores):.4f}, mean={np.mean(avg_impostor_scores):.4f}')
    
    # Plot and Compute Metrics
    roc_auc_committee, thresholds = plot_analysis(avg_genuine_scores, avg_impostor_scores, "Subject-Dependent Committee")
    d_prime_committee = calculate_d_prime(avg_genuine_scores, avg_impostor_scores)
    print(f'Subject-Dependent Committee: AUC = {roc_auc_committee:.4f}, d-prime = {d_prime_committee:.4f}')
    
    # EER Threshold
    fpr, tpr, thresholds = roc_curve(
        np.concatenate([np.ones(len(avg_genuine_scores)), np.zeros(len(avg_impostor_scores))]),
        np.concatenate([avg_genuine_scores, avg_impostor_scores]))
    eer_threshold = thresholds[np.argmin(np.abs(tpr - (1 - fpr)))]
    print(f'Committee EER threshold: {eer_threshold:.4f}')
    
    # Rank Identification Rates
    rank1, rank3 = compute_rank_rates(
        avg_enrollment_features, avg_verification_features, avg_enrollment_labels, avg_verification_labels, eer_threshold)
    print(f'Subject-Dependent Committee: Rank-1 = {rank1:.4f}, Rank-3 = {rank3:.4f}')
    
    # Subject-Independent Protocol
    print("\nSubject-Independent Protocol")
    train_subjects = list(range(1, 41))
    test_subjects = list(range(41, 51))
    all_indices = list(range(1, 16))
    
    train_dataset = GeorgiaTechDataset(DATA_DIR, train_subjects, all_indices, data_transforms['train'])
    test_dataset = GeorgiaTechDataset(DATA_DIR, test_subjects, all_indices, data_transforms['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Train Model
    model = get_model(num_classes=40).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
        {'params': model.layer3.parameters(), 'lr': LEARNING_RATE / 10},
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATE / 10}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    train_model(model, train_loader, criterion, optimizer, scheduler, device, NUM_EPOCHS)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'subject_independent.pth'))
    print(f"Saved subject-independent model weights to {os.path.join(WEIGHTS_DIR, 'subject_independent.pth')}")
    
    # Display Sample Validation Images
    print("\nDisplaying Sample Validation Images (Subject-Independent)")
    display_validation_images(model, test_loader, device, num_images=6)
    
    # Extract Features
    model.fc = nn.Identity()
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # Enrollment and Verification
    enrollment_features = []
    enrollment_labels = []
    verification_features = []
    verification_labels = []
    
    for subject in range(40, 50):
        subj_idx = np.where(test_labels == subject)[0]
        if len(subj_idx) >= 5:
            enrollment_features.append(test_features[subj_idx[0]])
            enrollment_labels.append(test_labels[subj_idx[0]])
            verification_features.extend(test_features[subj_idx[1:5]])
            verification_labels.extend(test_labels[subj_idx[1:5]])
    
    enrollment_features = np.array(enrollment_features)
    enrollment_labels = np.array(enrollment_labels)
    verification_features = np.array(verification_features)
    verification_labels = np.array(verification_labels)
    
    # Cosine Similarity Scores
    genuine_scores, impostor_scores = compute_cosine_scores(
        enrollment_features, verification_features, enrollment_labels, verification_labels)
    
    # Print Score Ranges
    print(f'Genuine scores: min={np.min(genuine_scores):.4f}, max={np.max(genuine_scores):.4f}, mean={np.mean(genuine_scores):.4f}')
    print(f'Impostor scores: min={np.min(impostor_scores):.4f}, max={np.max(impostor_scores):.4f}, mean={np.mean(impostor_scores):.4f}')
    
    # Plot and Compute Metrics
    roc_auc, _ = plot_analysis(genuine_scores, impostor_scores, "Subject-Independent")
    d_prime = calculate_d_prime(genuine_scores, impostor_scores)
    print(f'Subject-Independent: AUC = {roc_auc:.4f}, d-prime = {d_prime:.4f}')

if __name__ == '__main__':
    main()
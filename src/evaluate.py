import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

def plot_history(history):
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(history['train_loss'], label='Train Loss')
  plt.plot(history['val_loss'], label='Val Loss')
  plt.title('Loss over epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(history['train_acc'], label='Train Accuracy')
  plt.plot(history['val_acc'], label='Val Accuracy')
  plt.title('Accuracy over epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.legend()
  plt.tight_layout()
  plt.show()

def plot_confusion_matrix(model, dataloader, device, class_name_list):
    model.to(device)
    model.eval()
    predictions = []
    all_labels = []
    correct_test = 0
    total_test = 0

    with torch.no_grad():
       for inputs, labels in dataloader:
          inputs = inputs.to(device)
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)

          predictions.extend(predicted.cpu().numpy())
          all_labels.extend(labels.numpy())

          total_test += labels.size(0)
          correct_test += (predicted == labels).sum().item()

    cm = confusion_matrix(all_labels, predictions)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_name_list, yticklabels=class_name_list)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels') 
    plt.title('Confusion matrix')
    plt.show()       

    test_accuracy = (correct_test/ total_test) * 100.0
    print(f' - Model accuracy: {test_accuracy:.2f}%')

def plot_tsne(model, dataloader, device, class_names_list, n_iter=1000, perplexity=50.0):
    model.eval()
    all_features = []
    all_labels = []

    # Tạo một hook để lấy output của lớp trước lớp fully connected
    # Trong PaperModel, đó là output của self.avgpool sau khi flatten
    features_from_hook = []
    def hook_fn(module, input, output):
        # output ở đây là sau self.avgpool, có shape (batch_size, channels, 1, 1)
        # cần flatten nó thành (batch_size, channels)
        flattened_output = torch.flatten(output, 1)
        features_from_hook.append(flattened_output.detach().cpu().numpy())

    # Đăng ký hook vào lớp self.avgpool
    # Lưu ý: Nếu bạn thay đổi kiến trúc, bạn có thể cần thay đổi lớp để hook
    hook_handle = model.avgpool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # Chạy forward pass để hook được kích hoạt và features_from_hook được cập nhật
            _ = model(inputs)
            all_labels.extend(labels.cpu().numpy())

    hook_handle.remove() # Gỡ hook sau khi đã lấy xong features

    # Nối các batch features lại
    if features_from_hook:
        all_features = np.concatenate(features_from_hook, axis=0)
    else:
        print("Không có features nào được trích xuất. Kiểm tra lại hook hoặc dataloader.")
        return

    if len(all_features) == 0:
        print("Mảng features rỗng.")
        return

    print(f"Trích xuất được {all_features.shape[0]} features với {all_features.shape[1]} chiều.")

    # Giảm chiều bằng t-SNE
    tsne = TSNE(n_components=2, random_state=42, n_iter=n_iter, perplexity=perplexity, n_jobs=-1)
    tsne_results = tsne.fit_transform(all_features)

    # Vẽ

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap=plt.cm.get_cmap("jet", len(class_names_list)), alpha=0.7)
    plt.title('t-SNE visualization of learned features')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')

    # Tạo legend
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.9)
    if len(handles) == len(class_names_list):
         plt.legend(handles, class_names_list, title="Classes")
    else: # Trường hợp số lượng unique labels trong batch nhỏ hơn NUM_CLASSES
        unique_labels_in_data = sorted(list(set(all_labels)))
        filtered_class_names = [class_names_list[i] for i in unique_labels_in_data]
        if len(handles) == len(filtered_class_names):
            plt.legend(handles, filtered_class_names, title="Classes")
        else:
            print(f"Không thể tạo legend chính xác. Số lượng handles: {len(handles)}, số lượng class_names: {len(class_names_list)}, số lượng unique labels: {len(unique_labels_in_data)}")


    plt.show()





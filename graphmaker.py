import pandas as pd
import matplotlib.pyplot as plt

log_path = "training_logs/effnetv2_ft_log.csv" 
df = pd.read_csv(log_path)

# val vs epoch
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy", color="blue", marker="o")
plt.plot(df["epoch"], df["accuracy"], label="Training Accuracy", color="green", linestyle="--")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="red", marker="o")
plt.plot(df["epoch"], df["loss"], label="Training Loss", color="orange", linestyle="--")
plt.title("Validation & Training Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("EfficientNetV2 (fine tuned)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("training_plots/effnetv2_ft.png", dpi=300)
plt.close()

# # valloss vs epoch
# plt.figure(figsize=(8, 5))
# plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="red", marker="o")
# plt.plot(df["epoch"], df["loss"], label="Training Loss", color="orange", linestyle="--")
# plt.title("Validation & Training Loss vs Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("EfficientNetV2 (fine tuned) Loss")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig("training_plots/effnetv2_ft_loss.png", dpi=300)
# plt.close()

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Paths
# log_path = "training_logs/cnn_bigru_ft_log.csv"
# save_dir = "training_plots"
# os.makedirs(save_dir, exist_ok=True)

# # Load training log
# df = pd.read_csv(log_path)

# # Define epochs where LR was reduced (change these as per your training logs)
# lr_reduction_epochs = [7, 12, 16, 20, 24, 28]

# # --- Accuracy Plot ---
# plt.figure(figsize=(8, 5))
# plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy", color="blue", marker="o")
# plt.plot(df["epoch"], df["accuracy"], label="Training Accuracy", color="green", linestyle="--")

# # Add vertical lines for LR reduction
# for e in lr_reduction_epochs:
#     plt.axvline(x=e, color='red', linestyle='--', alpha=0.6)
#     plt.text(e, plt.ylim()[1]*0.05, f'LR↓@{e}', rotation=90, color='red', fontsize=8, ha='right')

# plt.title("Validation & Training Accuracy vs Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("CNN+GRU(fine tuned) Accuracy")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig(f"{save_dir}/cnngru_ft_acc.png", dpi=300)
# plt.close()

# # --- Loss Plot ---
# plt.figure(figsize=(8, 5))
# plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="red", marker="o")
# plt.plot(df["epoch"], df["loss"], label="Training Loss", color="orange", linestyle="--")

# # Add vertical lines for LR reduction
# for e in lr_reduction_epochs:
#     plt.axvline(x=e, color='red', linestyle='--', alpha=0.6)
#     plt.text(e, plt.ylim()[1]*0.05, f'LR↓@{e}', rotation=90, color='red', fontsize=8, ha='right')

# plt.title("Validation & Training Loss vs Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("CNN+GRU (fine tuned) Loss")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig(f"{save_dir}/cnngru_ft_loss.png", dpi=300)
# plt.close()

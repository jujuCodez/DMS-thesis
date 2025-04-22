import os
import torch
import timm

# -------------------------------
# Configuration
# -------------------------------
src_folder = r"J:\DMS_Thesis_CNN_4\trained_models\pth"
dst_folder = r"J:\DMS_Thesis_CNN_4\trained_models\torchscript"

# Create the destination folder if it doesn't exist
os.makedirs(dst_folder, exist_ok=True)

# -------------------------------
# Conversion Process
# -------------------------------
for filename in os.listdir(src_folder):
    if filename.endswith(".pth"):
        pth_path = os.path.join(src_folder, filename)
        
        # Assume file name format is "DMS_{model_name}.pth", so extract the model name
        base_name = os.path.splitext(filename)[0]  # e.g., "DMS_mobilenetv3_large_100"
        if base_name.startswith("DMS_"):
            model_name = base_name[4:]
        else:
            model_name = base_name
        
        print(f"Converting {filename} as model '{model_name}'...")
        
        # -------------------------------
        # 1. Define & Initialize the Model
        # -------------------------------
        # Here, 'pretrained' is set to False because we're loading custom weights
        model = timm.create_model(model_name, pretrained=False)
        
        # Reset the classifier if your trained model uses a different number of classes
        # Change the number below if needed:
        model.reset_classifier(5)
        
        # -------------------------------
        # 2. Load the Trained Weights
        # -------------------------------
        state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        # -------------------------------
        # 3. Convert the Model to TorchScript
        # -------------------------------
        dummy_input = torch.rand(1, 3, 224, 224)  # Adjust the input size if necessary
        scripted_model = torch.jit.trace(model, dummy_input)
        
        # -------------------------------
        # 4. Save the TorchScript Model
        # -------------------------------
        output_filename = base_name + ".pt"
        output_path = os.path.join(dst_folder, output_filename)
        scripted_model.save(output_path)
        print(f"Saved TorchScript model to {output_path}\n")

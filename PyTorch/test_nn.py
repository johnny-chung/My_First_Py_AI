import torch
from SimpleNN import SimpleNN
from CNN import MyCNN

# Example: Assuming your input is an 8x8 2D array
input_data = torch.tensor([
            [0,  0,  0,  0,  1, -1,  0,  0],
            [0,  0,  0,  0,  1,  1, -1,  1],
            [1,  1,  1,  1,  1, -1,  1,  1],
            [0, -1, -1,  1, -1, -1,  1,  1],
            [0,  0,  1,  1,  1,  1,  0,  1],
            [-1,-1, -1, -1, -1, -1, -1, -1],
            [0,  0,  0,  0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0]
    
            # [0,  0,  0,  0,  0,  0,  0,  0],
            # [0,  0,  0,  0,  0,  0,  0,  0],
            # [0,  0,  0,  0,  0,  0,  0,  0],
            # [0,  0,  0, -1, -1,  0,  0,  0],
            # [0,  0,  0, -1,  1,  0,  0,  0],
            # [0,  0,  0,  0, -1,  0,  0,  0],
            # [0,  0,  0,  0,  0,  0,  0,  0],
            # [0,  0,  0,  0,  0,  0,  0,  0]

        ], dtype=torch.float32)

# Reshape the input to match the expected shape (batch_size, channels, height, width)
input_data = input_data.unsqueeze(0).unsqueeze(0)  # Assuming the model expects a single-channel input

# Load the model and set it to evaluation mode
model = MyCNN()
model.load_state_dict(torch.load('trained_reversi_model_2.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    output = model(input_data)

# Post-process the output if necessary
# Example: Convert the output to a 2D array
    

# rounded_predictions = torch.round(output)
# output_array = rounded_predictions.view(8, 8).numpy()


print(output)

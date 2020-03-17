import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description = "\t\t\t\tDescription:\n"
                                "--------------------------------------------------------------------------\n"
                                "\t\t   Run predictions on numbers in images\n\n" 
                                "---------------------------------------------------------------------------",
                                formatter_class = RawTextHelpFormatter)

args = parser.parse_args()

def predict_numbers(demonstration = False):
    """
    This is a do it all in one function for loading a trained model for predicting
    handwritten digits from pictures and then renaming those pictures.
    
    > demonstration - indicate whether you want to see what files are being worked with and predictions on them.
    """
    
    # (0) Make all the required imports
    import torch                              
    import torch.nn as nn                                       
    from torchvision import transforms 
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import os
    from os import walk
    from os.path import join
    print("[X] Imports completed")
    
    # (1) Prepare and load the model
    
    # Load the device the user will use
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    class Network(nn.Module):
        def __init__(self):                                             
            super(Network, self).__init__()                             
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels = 1,
                                       out_channels = 16,
                                       kernel_size = 5,
                                       stride = 1,
                                       padding = 2),                        
                nn.ReLU(),                                                
                nn.MaxPool2d(kernel_size=2, 
                             stride=2)                                    
                )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels = 16,
                                       out_channels = 32,
                                       kernel_size = 5,
                                       stride = 1,
                                       padding = 2),                        
                nn.ReLU(),                                                
                nn.MaxPool2d(kernel_size = 2,
                             stride = 2)                                    
                )

            self.fc1 = nn.Linear(in_features = 7*7*32,                  
                                 out_features = 10)                    

        def forward(self, t):
            out = self.conv1(t)
            out = self.conv2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            return out

    # Load the model into an executable form
    model = Network().to(device)
    print("[X] Network prepared")
    
    # (2) Load weights into the model that were trained
    model.load_state_dict(torch.load("model.ckpt", map_location=torch.device('cpu')))
    print("[X] Network weights updated")
    
    # (3) Prepare the images pre-processing function
    def images_preprocess(path):
        """
        An image is preprocessed and returned in a form that 
        the NN model understands.
        """
        img = cv2.imread(path, 0)                      
        img_inv = cv2.bitwise_not(img)                 
        img_inv_res = cv2.resize(img_inv, (28, 28))    
        img_inv_res_tensor = torch.tensor(img_inv_res)
        
        if demonstration == True:
            plt.imshow(img_inv_res_tensor, cmap = "gray")
            plt.show()
        return img_inv_res_tensor
    print("[X] Preprocessing of Images prepared")

    # (4) Prepare image accumulating variable (users' images)
    storage = "ToPredict/"  
    images = []     

    files = []
    for (dirpath, dirnames, filenames) in walk(storage):
        files.extend(filenames)

    for file in files:
        image = images_preprocess(join(storage, file))
        images.append(image)
    print("[X] Images Loaded")
        
    # (5) Prepare the data loader to the model
    loader_images = torch.utils.data.DataLoader(dataset = images,
                                                batch_size = 16,
                                                shuffle = False)
    print("[X] Data loader prepared")
    
    # (6) Prepare the predictions making function
    def prediction(model, data_loader):
        """
        Make predictions with the model.
        """
        model.eval()
        all_predictions = []
        final_predictions = []

        with torch.no_grad():  

            # Predict one by one
            for images in data_loader:
                images = images.reshape(-1, 1, 28, 28).to(device, dtype=torch.float) 
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.append(predicted.cpu().numpy())

            for entry in all_predictions:
                for value in entry:
                    final_predictions.append(value)
                    
            if demonstration == True:
                print(final_predictions)
                
            return final_predictions
    print("[X] Function for predictions prepared")
    
    # (7) Make the actual predictions
    predicted = prediction(model = model,
                           data_loader = loader_images)
    print("[X] Predictions made")
    
    # (8) Rename the files
    counter = 0
    for (dirpath, dirnames, filenames) in walk("ToPredict"):
        for file in filenames:
            os.rename("ToPredict/" + str(filenames[counter]), "ToPredict/" + "Case" + str(counter) + "_" + str(predicted[counter]) + ".png")
            counter += 1
    print("[X] Files renamed")

if __name__ == "__main__":
    print("[X] Execution Started")
    print("-------------------------------------------")
    predict_numbers()
    print("-------------------------------------------")
    print("[X] Execution Finished")
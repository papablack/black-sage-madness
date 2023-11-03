
> Prepare code in node TS for preparing dataset to train stable diffusion model. The code must push the dataset to python script. Make python execution in separate service file


To prepare a dataset for training a stable diffusion model and push it to a Python script, you would typically follow these steps:

1.  Collect and preprocess the dataset in Node.js.
2.  Serialize the dataset into a format that can be consumed by the Python script.
3.  Execute the Python script from Node.js, passing the dataset as an argument or through standard input.

Below are three full example files for a hypothetical setup where you have a dataset of images that you want to tag and use to train a Stable Diffusion model. These examples are based on the assumption that you have a pre-trained model file and a corresponding Python class that defines the model architecture.

**src/service/DatasetProcessor.ts:**

typescript

```typescript
// DatasetProcessor.ts
import * as fs from 'fs';
import * as path from 'path';

export class DatasetProcessor {
  private dataset: any[];

  constructor() {
    this.dataset = [];
  }

  public async prepareDataset(): Promise<void> {
    // Your dataset preparation logic here
    // This is just a placeholder logic
    this.dataset = [
      // ... your dataset objects
    ];
    // ... more preparation logic
  }

  public getDataset(): any[] {
    return this.dataset;
  }

  public writeDatasetToFile(filePath: string): void {
    const dataString = JSON.stringify(this.dataset);
    fs.writeFileSync(filePath, dataString);
  }
}
```

**src/service/PythonCaller.ts:**

typescript

```typescript
// PythonCaller.ts
import { spawn } from 'child_process';
import DatasetProcessor from './DatasetProcessor';

const PythonCaller = {
  callPythonService(scriptPath: string, stdinData: any): void {
    return new Promise(function (resolve, reject){
      const pythonProcess = spawn('python', [scriptPath]);

      pythonProcess.stdin.write(JSON.stringify(stdinData.getDataset()));
      pythonProcess.stdin.end();

      pythonProcess.stdout.on('data', function(data) {
        console.log(`Python Service Output: ${data}`);
        resolve(data);
      });

      pythonProcess.stderr.on('data', function(data) {
        console.error(`Python Service Error: ${data}`);
        reject(data);
      });

      pythonProcess.on('close', function(code) {
        console.log(`Python Service exited with code ${code}`);
      });
    });
  }
}

export default PythonCaller;

// src/index.ts
async function main() {  
  await DatasetProcessor.prepareDataset();  
  PythonCaller.callPythonService('../pyscripts/sd_training.py', [datasetProcessor]);
}

main();
```
**pyscripts/_custom_dataset.py**
```python
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class CustomDataset(Dataset):
    def __init__(self, dataset_json, img_dir, transform=None):
        """
        Args:
            annotations_file (string): Path to the json file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = dataset_json
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx]['image_id'])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx]['annotations']
        if self.transform:
            image = self.transform(image)
        return image, label
```


**pyscripts/sd\_training.py:**

python

```python
# sd_training.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from ._custom_dataset import CustomDataset
import json
import sys

def train_stable_diffusion_model(dataset, model, preTrainedPath):      
    # Load the pre-trained weights
    model.load_state_dict(torch.load(preTrainedPath))

    if torch.cuda.is_available():
        model.cuda()

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Example loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    # Training loop
    num_epochs = 5  # Example number of epochs
    for epoch in range(num_epochs):
        for i, data in enumerate(dataset):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataset)}], Loss: {loss.item()}')

    return {
      'success': True,
      'loss_data': losses 
    }

class SD2CustomModel(nn.Module):
  def __init__(self):
    super(SD2CustomModel, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Calculate the size of the flattened features after the final pooling layer
    # This will depend on the number of pooling layers and their kernel sizes/strides
    self.flattened_size = 256 * (512 // (2 ** num_pooling_layers)) ** 2
    self.fc1 = nn.Linear(self.flattened_size, 1024)


  def forward(self, x):      
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))

    # Flatten the tensor for the fully connected layer
    x = x.view(-1, self.flattened_size)
    x = F.relu(self.fc1(x))

    # Add more layers or a final classification/regression layer if needed
    return x 

#Execute script

dataset_json = sys.stdin.read()
dataset_data = json.loads(dataset_json)
preTrainedPath = './pre-trained/sd20.pth'
preTrainedFilename = preTrainedPath.split('/')[-1]
preTrainedExt = preTrainedFilename.split('.')[-1]
preTrainedModelname = preTrainedFilename.split('.')[-2]
the_model = SD2CustomModel('TrainingSDModel', preTrainedModelname)

# Convert dataset to a format suitable for training
# This is a placeholder; you'll need to replace it with your actual data loading logic
dataset = CustomDataset(dataset_json=dataset_data, img_dir='path/to/images', transform=transforms.ToTensor())
# Call the training function
print(json.dumps(train_stable_diffusion_model(dataset, the_model, preTrainedPath), indent=4))
```

Please note that these files are placeholders and contain example code that will need to be adapted to your specific use case. The `DatasetProcessor` class in `DatasetProcessor.ts` should contain logic to prepare your dataset, and `sd_training.py` should be modified to include the actual data loading and model training logic based on the pre-trained Stable Diffusion model you have.
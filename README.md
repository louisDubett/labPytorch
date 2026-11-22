# install
## Create a virtual environment 
`python -m venv .venv`

## Set up the virtual environment & data

###  on Windows
`.venv\Scripts\activate`

### on macOS and Linux
`source .venv/bin/activate`

### install all packages
`pip install -r requirements.txt` or manually with `pip install torch torchvision matplotlib` 

### Get the dataset from hugging face
[labGlass](https://huggingface.co/datasets/louisDubett/labGlass/tree/main)

# use the model locally on Windows
`run_local`

# use the model with a browser (localhost:8080) on Windows
`web.bat`

# Generate the model
activate the virtual environment like above ('myenv\Scripts\activate')
run the command to generate model.pth
`python image_classifier.py -train`

# test the model with 1 image
`python image_classifier.py -image <image_file_name>`

# evaluate the model with test data
`python evaluate_model.py -config config-accuracy-test-easy.xml`



The evaluation produce a xml result file (with for each image OK/not OK) and a confusion matrix (named confusion.png)

# Title Generator - T5 
An application that generates paper titles given an abstract. A template for projects involving T5 LLMs. 

### Installation:

To get started with this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-project.git
   cd your-project
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage:

1. Fetching data from arxiv
```bash
python scripts/fetch_arxiv_data.py
```

2. Fine-tuning the T5 model
```bash
python scripts/finetuning.py
```

### Sagemaker Deployment

Run the "deployment/deployment_flow.ipynb" notebook to deploy the model to Sagemaker. 



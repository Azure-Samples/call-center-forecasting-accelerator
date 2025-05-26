# <img src="./utils/media/ai-foundry.jpg" alt="Azure Foundry" style="width:60px;height:60px"/> 📞 Advanced Machine Learning Forecasting for Call Center Operations
## 🔧 1. Prerequisites

+ [azd](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd), used to deploy all Azure resources and assets used in this sample.

+ [azure functions core tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=windows%2Cisolated-process%2Cnode-v4%2Cpython-v2%2Chttp-trigger%2Ccontainer-apps&pivots=programming-language-csharp)

+ [PowerShell Core pwsh](https://github.com/PowerShell/powershell/releases) if using Windows

+ [Python 3.11](https://www.python.org/downloads/release/python-3110/)

## 🔧 2. Infrastructure Creation

This sample uses [`azd`](https://learn.microsoft.com/azure/developer/azure-developer-cli/) and a bicep template to deploy all Azure resources:

1. Login to your Azure account: `azd auth login`

2. Create an environment: `azd env new`

3. Place documents for testing inside [data](./data/) folder 

4. Run `azd up`.

   + Choose your Azure subscription.
   + Enter a region for the resources.

   The deployment creates multiple Azure resources and runs multiple jobs. It takes several minutes to complete. The deployment is complete when you get a command line notification stating "SUCCESS: Your up workflow to provision and deploy to Azure completed."

## 📦 Installation

1. **Set up Your Environment**  
   If you're using Azure, select your environment as shown in the image below:  
   ![Select Azure Environment](./utils/media/env-azure.png "Select your Azure environment")

2. **Clone the Repository**  
   Run the following commands in your terminal:
   ```bash
   git clone https://github.com/your-org/gbbai-o1-reasoning-over-ml.git
   cd gbbai-o1-reasoning-over-ml
   ```

3. **(Optional) Create and Activate a Python Virtual Environment**  
   If you're not in Azure (or prefer to work locally), create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate or venv/Scripts
   ```

4. **Install Dependencies**  
   Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

*This process can be executed via the terminal in your AML Studio or through VSCode connected to your VM compute.*

![Execute throught terminal](./utils/media/terminal-azure.png "Select your Azure environment")

## Environment Variables Setup

To ensure proper execution of the project, append the following code to set up key environment variables. These variables are essential for connecting with your Azure resources:

```python
subscription=
resource_group=
workspace=
datastore_name = 
path_on_datastore = 
storage_account_name = 

```

Make sure to define these variables in your local environment or include them in your configuration settings before running the application.

## 🔧 3. Context

In today’s fast-paced service environment, forecasting for call centers is crucial. Leveraging a machine learning model allows the application to predict call volumes months in advance. This capability supports better workforce planning by informing decisions on timely hiring and staff reallocation, ensuring that enough personnel are available during peak periods. Moreover, early predictions enable the organization to optimize resources, prepare targeted training sessions, and implement proactive strategies, ultimately enhancing customer satisfaction and operational efficiency.



## Use of the notebooks

These two notebooks are designed to work together as parts of your overall forecasting solution. Here’s how you can use them in your workflow:

Data Preparation & Analysis Notebook (01-data-prep-analysis-detailed.ipynb):
• Data Ingestion:
  – Load your raw CSV using MLTable so that you can track, version, and share your data asset.
  – Convert column types manually if needed (e.g. converting date columns to datetime).
• Data Cleaning:
  – Remove unneeded identifiers, duplicate rows, and unnecessary columns.
  – Normalize categorical data (for example, grouping and encoding closure reasons).
• Feature Engineering:
  – Create new features like time_to_resolve, holiday flags, and aggregated daily call counts.
  – Adjust the target variable by aggregating call counts over a future 30-day window using rolling windows.
• Exploratory Analysis:
  – Visualize trends such as daily and monthly call counts.
  – Check for seasonality and data distribution.
• Saving Data Assets:
  – Save your processed data as MLTable assets (the “silver” layer) to be used downstream.

Model Deployment & Evaluation Notebook (03-deploy-automl-forecasting-call-center-mlflow.ipynb):
• Connect to Azure ML Workspace & MLFlow:
  – Set up your Azure ML workspace and establish a connection using the default Azure credentials.
  – Retrieve the AutoML trial’s best run and its metrics via MLFlow.
• Model Preparation:
  – Download the best model along with its environment and related artifacts. • Endpoint Creation & Deployment:
  – Register the model and environment.
  – Create a batch endpoint and deploy the model so that it can be used for batch inferencing.
• Forecast vs. Actuals Analysis:
  – Invoke the deployed model endpoint on test data.
  – Download the forecast results and join them with historical data.
  – Visualize the predictions against the actual values, including plotting prediction intervals.

Workflow Summary:
– First, run the data preparation notebook to understand and prepare your dataset, create consistent MLTable assets, and perform feature engineering.
– Then, use the deployment notebook to retrieve the best model from an AutoML run, deploy it using Azure ML batch endpoints, run inferencing on test data, and analyze the forecasts.

By separating these concerns, you ensure both a robust data pipeline and a smooth model deployment process, which together support efficient forecasting and analysis.

## Data 

## Data Visualization

Below is an overview of the call center data, illustrating the raw data input and its transformation via the pipeline:

![Data Pipeline Overview](./media/utils/data.png "Overview of the Call Center Data Pipeline")
i have a dataframe which contains data from a call center. Each row represents the opening call opened by a customer, i want to predict the volume of calls next month, how can i adjust this data to be able to do this

⚠️ Attention: In data_call_center.csv, the "raw" data is a MOCK dataset provided solely for demonstration purposes and to run our presented pipelines. Please adjust your data pipeline accordingly before applying it to real call center data.

### Use the data pipeline


The idea behind this pipeline (core->data_preparation->pipeline) is to automatically perform all the data preparation steps you need for your forecasting model. Instead of manually cleaning, aggregating, and feature‐engineering your raw data every time, you define the transformations once and then apply them via a scikit‑learn pipeline.

Configure the Pipeline:
First, you provide a configuration (an instance of FeatureEngineeringConfig) that outlines which data‐preparation steps (e.g. dropping columns, closure reason processing, aggregating daily features, computing the next 30 days target, and adding a holiday flag) should run. The DataPrepPipelineBuilder reads this configuration and builds a scikit‑learn Pipeline containing your transformation steps.

Using the Pipeline:
Once the pipeline is built, you call its fit and transform methods. For example, if your raw call center data is in a CSV file, you can instantiate the DataVolumePreparation transformer with the CSV path. That transformer will then:

Load and clean the raw data (convert dates, fill missing values, compute time-to-resolve, etc.).
Process the closure reasons (grouping similar closures into common categories).
Aggregate the data by day, compute daily call counts, and create a target variable ("target_next_30days") that sums the call counts over the next 30 days.
Add additional features like the Brazilian holiday flag.
Automatic Data Operations:
With these steps automated in the pipeline, every time you run your pipeline you get a preprocessed DataFrame that is ready to be used for training your forecasting model. The advantage is consistency and reduced manual effort, especially when dealing with large datasets and complex transformations

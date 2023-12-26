# AD-Project
# Installation
To run this Streamlit Data Analysis project, follow these steps:

# 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```
Replace **"your-username"** and **"your-repository"** with your [GitHub](https://github.com/) username and the name of your repository.

# 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
```
Activate the virtual environment:

## On Windows:

```bash
venv\Scripts\activate
```

## On macOS/Linux:

```bash
source venv/bin/activate
```
# 3. Install Required Libraries
Install the necessary Python libraries using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install -r requirements.txt
```
The **"requirements.txt"** file should contain the following:

```bash
# Core libraries
certifi==2022.12.5
numpy==1.21.4
pandas==1.3.3

# Visualization libraries
plotly==5.3.1
matplotlib==3.4.3
seaborn==0.11.2

# Data manipulation libraries
scikit-learn==0.24.2

# Statistical analysis libraries
scipy==1.7.3
statsmodels==0.13.1

# Streamlit
streamlit==1.8.0

# Other libraries
requests==2.26.0

```
# 4. Run the Streamlit Application
Now that the dependencies are installed, you can run the Streamlit application. Make sure you are in the project directory and the virtual environment is activated.

```bash
streamlit run Data-Analysis-Project.py

```
Replace **"Data-Analysis-Project.py"** with the name of your Streamlit application file.

This will start a local development server, and you can access the application in your web browser at [localhost](http://localhost:8501).

# Usage
Explore the user-friendly and interactive platform allowing users to easily load, process, analyze, visualize tabular data and perform analysis testing.

# Contributors


   * TAGHZOUT Imane.

   * BAALI Ghizlane.

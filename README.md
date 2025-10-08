# SSD Exercise 2

Tools for basic database interaction and exploratory data analysis (EDA).

## Setup

```bash
cd "/Users/kareloven/Projects/SSD/Exercise 2"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Local mysql setup
Done on Macbook Pro with Apple silicone

1. Install Docker desktop
2. Run ```docker pull mysql:latest```
3. Run ```docker run -p 3306:3306 --name <container-name> -e MYSQL_ROOT_PASSWORD="<your-password> -e MYSQL_DATABASE=<db-name> -d mysql:latest"```

## Run EDA

```bash
python eda_porto.py               # saves outputs in eda_reports/
# or use the notebook
jupyter notebook eda_porto.ipynb
```

## DB Example
See `example.py` and `DbConnector.py` for simple MySQL usage.

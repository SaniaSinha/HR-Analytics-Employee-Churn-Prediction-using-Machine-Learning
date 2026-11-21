
import os
import certifi

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "hr_churn_db")

client = None
database = None
predictions_collection = None

if MONGODB_URI:
    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=15000,
        tlsCAFile=certifi.where()
    )

    database = client[DATABASE_NAME]
    predictions_collection = database["predictions"]
_collection = database["predictions"]
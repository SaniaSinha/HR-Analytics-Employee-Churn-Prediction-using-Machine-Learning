
from pydantic import BaseModel, Field

class EmployeeData(BaseModel):
    satisfaction_level: float = Field(ge=0, le=1)
    last_evaluation: float = Field(ge=0, le=1)
    number_project: int = Field(ge=1)
    average_monthly_hours: int = Field(ge=0)
    time_spend_company: int = Field(ge=0)
    Work_accident: int = Field(ge=0, le=1)
    promotion_last_5years: int = Field(ge=0, le=1)
    department: str
    salary: str

class PredictionRequest(EmployeeData):
    experiment: str = Field(
        description="One of the 21 experiment names, e.g. BASE - Random Forest"
    )
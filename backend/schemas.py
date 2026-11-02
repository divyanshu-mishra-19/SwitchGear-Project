"""
CircuitGuard AI - API Data Schemas (Pydantic Models)
Defines data structures, request schemas, and responses for the FastAPI backend.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CircuitBreakerInput(BaseModel):
    # Categorical features
    Product_variant: str = Field(default="A", description="Utility Fleet ID (A, B, C, D, E)")
    Breaker_status: str = Field(default="Closed", description="Switch State (Closed, Open)")
    Heater_status: str = Field(default="Off", description="Cabinet Heater State (On, Off)")
    Last_trip_coil_energized: str = Field(default="TC1", description="Trip Coil (TC1, TC2)")

    # Numerical features
    SF6_density: float = Field(default=6.05, ge=4.0, le=8.0, description="SF6 Gas Density (kg/m³)")
    SF6_temperature: float = Field(default=20.0, ge=-20.0, le=80.0, description="SF6 Gas Temperature (°C)")
    SF6_dew_point: float = Field(default=19.5, ge=-40.0, le=60.0, description="SF6 Dew Point Moisture (°C)")
    Motor_run_time: float = Field(default=5.5, ge=0.1, le=30.0, description="Mechanism Charging Motor Run Time (s)")
    Motor_runs_per_day: int = Field(default=1, ge=0, le=20, description="Charging Motor Cycles per Day")
    Motor_I2T: float = Field(default=7.2, ge=0.0, le=50.0, description="Motor Thermal Stress (A²·s)")
    Line_voltage: float = Field(default=120000.0, ge=50000.0, le=200000.0, description="Grid Operating Line Voltage (V)")
    Ambient_cabinet_temperature: float = Field(default=58.5, ge=-20.0, le=140.0, description="Cabinet Ambient Temperature (°C)")
    Days_since_last_operation: Optional[float] = Field(default=30.0, ge=0.0, le=365.0, description="Days Since Last Switch Operation")
    Contact_life: float = Field(default=33000.0, ge=1000.0, le=150000.0, description="Contact Wear Life Rating Points")
    Clearing_time: Optional[float] = Field(default=0.045, ge=0.005, le=0.5, description="Fault Current Interrupt Time (s)")
    Open_latch_time: float = Field(default=0.095, ge=0.01, le=0.5, description="Open Mechanism Latch Time (s)")
    Close_latch_time: float = Field(default=0.098, ge=0.01, le=0.5, description="Close Mechanism Latch Time (s)")
    Opening_travel_time: float = Field(default=0.078, ge=0.01, le=0.5, description="Contact Travel Duration Open (s)")
    Closing_travel_time: float = Field(default=0.079, ge=0.01, le=0.5, description="Contact Travel Duration Close (s)")
    Total_operation_count: int = Field(default=4200, ge=0, le=20000, description="Cumulative Switching Operations")
    Non_fault_operation_count: int = Field(default=3650, ge=0, le=20000, description="Normal Switching Cycles")
    Fault_operations_count: int = Field(default=550, ge=0, le=10000, description="Fault Interruption Cycles")
    Accumulated_motor_run_time: Optional[float] = Field(default=420.0, ge=0.0, le=10000.0, description="Total Motor Charging Time (s)")
    Heater_current: Optional[float] = Field(default=0.0, ge=0.0, le=10.0, description="Current Drawn by Heater (A)")
    Open_average_velocity: Optional[float] = Field(default=3.1, ge=0.1, le=15.0, description="Contact Velocity Open (m/s)")
    Close_average_velocity: Optional[float] = Field(default=3.1, ge=0.1, le=15.0, description="Contact Velocity Close (m/s)")
    Restrike_occurrence: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Arc Restrike Occurrence Flag (0 or 1)")
    Open_operation_arcing_time: Optional[float] = Field(default=0.024, ge=0.001, le=0.5, description="Arc Duration During Opening (s)")

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "Product_variant": "A",
                "Breaker_status": "Closed",
                "SF6_density": 6.05,
                "SF6_temperature": 20.0,
                "SF6_dew_point": 19.5,
                "Motor_run_time": 6.5,
                "Motor_runs_per_day": 1,
                "Motor_I2T": 8.5,
                "Heater_status": "Off",
                "Line_voltage": 120000.0,
                "Ambient_cabinet_temperature": 65.0,
                "Days_since_last_operation": 30.0,
                "Contact_life": 33000.0,
                "Last_trip_coil_energized": "TC1",
                "Clearing_time": 0.045,
                "Open_latch_time": 0.11,
                "Close_latch_time": 0.11,
                "Opening_travel_time": 0.08,
                "Closing_travel_time": 0.08,
                "Total_operation_count": 4200,
                "Non_fault_operation_count": 3600,
                "Fault_operations_count": 600,
                "Accumulated_motor_run_time": 400.0,
                "Heater_current": 0.5,
                "Open_average_velocity": 3.0,
                "Close_average_velocity": 3.0,
                "Restrike_occurrence": 0.0,
                "Open_operation_arcing_time": 0.025
            }
        }
    }

class PredictionRequest(BaseModel):
    features: CircuitBreakerInput
    model_name: Optional[str] = Field("TabNet (Pretrained + Fine-Tuned)", description="Model selection")

class TopFeatureContribution(BaseModel):
    feature: str
    importance_score: float
    reading_value: Any
    unit: str

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    key_input_summary: Dict[str, Any]
    top_influential_features: List[TopFeatureContribution]
    explanation: str
    disclaimer: str

class BatchPredictionItem(BaseModel):
    row_index: int
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    total_records: int
    model_used: str
    predictions: List[BatchPredictionItem]
    maintenance_required_count: int
    no_maintenance_count: int

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    available_models: List[str]
    device: str
    version: str

class ModelInfoResponse(BaseModel):
    project_name: str
    primary_model: str
    transfer_learning_approach: str
    target_classes: Dict[str, str]
    input_features_count: int
    test_set_evaluation: Dict[str, Any]
    available_models: List[str]


from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any

class LoanApplicationRequest(BaseModel):
    """
    Validated input schema for an incoming loan application.
    """
    model_config = ConfigDict(extra="ignore")

    amt_income_total: float = Field(..., gt=0, description="Total annual income in USD", examples=[75000.0])
    amt_credit: float = Field(..., gt=0, description="Loan principal amount requested", examples=[250000.0])
    amt_annuity: float = Field(..., gt=0, description="Monthly/annual loan payment annuity", examples=[12500.0])
    days_birth: int = Field(..., lt=0, description="Age in negative days from application", examples=[-14600])
    days_employed: int = Field(..., description="Days employed (negative days, or 365243 for pensioner)", examples=[-2100])
    ext_source_1: Optional[float] = Field(None, ge=0.0, le=1.0, description="Bureau Score 1")
    ext_source_2: Optional[float] = Field(None, ge=0.0, le=1.0, description="Bureau Score 2")
    ext_source_3: Optional[float] = Field(None, ge=0.0, le=1.0, description="Bureau Score 3")
    region_rating_client: int = Field(2, ge=1, le=3, description="Regional rating (1=Best, 2=Medium, 3=Worst)")


class AdverseActionReason(BaseModel):
    feature: str
    shap_value: float
    reason_code: str


class LoanApplicationResponse(BaseModel):
    """
    Production decision response.
    """
    decision: str = Field(..., description="'APPROVED' or 'REJECTED'")
    default_probability: float = Field(..., description="Calibrated default probability P(Default|X)")
    threshold_applied: float = Field(..., description="Profit-maximizing rejection threshold")
    risk_tier: str = Field(..., description="'LOW_RISK', 'MEDIUM_RISK', or 'HIGH_RISK'")
    adverse_action_reasons: List[AdverseActionReason] = Field(
        default_factory=list,
        description="Top legal reasons if loan was rejected (FCRA/ECOA compliant)"
    )


class DriftCheckRequest(BaseModel):
    """
    Streaming batch of recent applications for drift monitoring.
    """
    applications: List[LoanApplicationRequest]


class DriftCheckResponse(BaseModel):
    overall_status: str = Field(..., description="'GREEN', 'YELLOW', or 'RED'")
    max_psi: float
    feature_metrics: Dict[str, Any]
    alert_triggered: bool

from pydantic import BaseModel, Field
from typing import Optional

class InvoiceExtractionBase(BaseModel):
    vendor_name: Optional[str] = Field(
        description="The name of the vendor, supplier, or billing entity."
    )
    vendor_address: Optional[str] = Field(
        description="The full physical address of the vendor."
    )
    amount_total_gross: Optional[str] = Field(
        description="The final total amount of the invoice, including all taxes."
    )
    date_issue: Optional[str] = Field(
        description="The date the invoice was issued or created. Format as YYYY-MM-DD if possible."
    )

# The "Thinking" schema extends the base schema by adding a scratchpad at the top
class InvoiceExtractionWithReasoning(InvoiceExtractionBase):
    reasoning_process: str = Field(
        description="Think step-by-step. Analyze the visual layout, locate the vendor details, find the date, and identify the total amount. Write your visual search process here before filling out the remaining fields."
    )
from pydantic import BaseModel, Field
from typing import Optional

# We define a single class to strictly enforce the order of keys in the JSON schema.
class InvoiceExtractionWithReasoning(BaseModel):
    # 1. THIS MUST BE FIRST. The model must think before it extracts.
    reasoning_process: str = Field(
        description=(
            "Internal scratchpad. Follow these steps quickly: "
            "1. Note the vendor name and address. "
            "2. Note the date. "
            "3. Scan the numbers. Identify Subtotal vs Tax vs Final Total. "
            "Write your findings here before filling out the exact fields below."
        )
    )
    
    vendor_name: Optional[str] = Field(
        description="The name of the vendor, supplier, or billing entity."
    )
    
    vendor_address: Optional[str] = Field(
        description="The full physical address of the vendor."
    )
    
    # 2. Strict constraint: min_length=1 completely forbids empty strings ("").
    invoice_total: str = Field(
        min_length=1, 
        description=(
            "The final total monetary amount of the invoice. "
            "Look for 'Total Due', 'Amount Due', or 'Grand Total'. "
            "Extract the exact numerical string."
        )
    )
    
    date_issue: Optional[str] = Field(
        description=(
            "The date the invoice was issued or created. Format as YYYY-MM-DD if possible."
            "CRITICAL: Output the raw date string ONLY. Do not include parentheses, "
            "notes, explanations, or corrections."
            )
    )

# Keeping the Base class purely so your eval_runner.py imports don't break
# if you toggle ENABLE_THINKING = False in the config.
class InvoiceExtractionBase(BaseModel):
    vendor_name: Optional[str] = Field(description="The name of the vendor, supplier, or billing entity.")
    vendor_address: Optional[str] = Field(description="The full physical address of the vendor.")
    invoice_total: str = Field(min_length=1, description="The final total amount of the invoice.")
    date_issue: Optional[str] = Field(description="The date the invoice was issued or created.")
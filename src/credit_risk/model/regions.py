"""US Census regions, used only to monitor outcomes by region (state is never a model input)."""
REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA".split(), "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD".split(),
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX".split(), "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA".split(),
}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states}

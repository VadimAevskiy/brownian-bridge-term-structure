# Data Dictionary

## Source

The dataset was originally provided by **Jasper Lund** (Copenhagen Business School) and used in Lund (1999). It contains weekly interest rate spreads between the Italian lira (ITL) and the German mark (DEM) for money market and swap instruments.

---

## Files

### `data/ITL.mat` (MATLAB format)

Contains a single matrix `emuweek` of shape (405, 11):
- **Row 0**: Header row with maturity values in years [0.083, 0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 0]
- **Rows 1--404**: Weekly data from 21 November 1990 to 12 August 1998

Also contains `textdata` (405, 1) with date strings in DD.MM.YYYY format.

### `data/ITL_DEM_data.xlsx` (Excel format)

Same data in spreadsheet format. Column layout matches the MATLAB matrix.

---

## Column Definitions (emuweek matrix)

| Column | Maturity | Type | Unit | Description |
|--------|----------|------|------|-------------|
| 0 | 1 month (0.083 yr) | Money market | % p.a. | ITL-DEM 1-month rate spread |
| 1 | 3 months (0.25 yr) | Money market | % p.a. | ITL-DEM 3-month rate spread |
| 2 | 6 months (0.5 yr) | Money market | % p.a. | ITL-DEM 6-month rate spread |
| 3 | 1 year | Money market | % p.a. | ITL-DEM 1-year rate spread |
| 4 | 2 years | Swap rate | % p.a. | ITL-DEM 2-year swap spread |
| 5 | 3 years | Swap rate | % p.a. | ITL-DEM 3-year swap spread |
| 6 | 4 years | Swap rate | % p.a. | ITL-DEM 4-year swap spread |
| 7 | 5 years | Swap rate | % p.a. | ITL-DEM 5-year swap spread |
| 8 | 7 years | Swap rate | % p.a. | Not used in estimation |
| 9 | 10 years | Swap rate | % p.a. | Not used in estimation |
| 10 | -- | -- | % p.a. | datataustar (auxiliary variable) |

---

## Time Dimensions

| Window | Start | End | Rows (MATLAB) | Rows (Python) | Observations |
|--------|-------|-----|---------------|---------------|--------------|
| Full dataset | 21.11.1990 | 12.08.1998 | 2 -- 405 | 1 -- 404 | 404 |
| Model subset | 27.12.1995 | 12.08.1998 | 268 -- 405 | 267 -- 404 | 138 |
| Estimation window | 20.08.1997 | 12.08.1998 | 354 -- 405 | 353 -- 404 | 52 |

**Frequency**: Weekly (Wednesdays)

**Why row 268?** The model requires a fixed time horizon T = 155 weeks until the EMU date (1 January 1999). The data subset starting 27 December 1995 is approximately 155 weeks before that date.

**Why row 87 (within the subset)?** The first 86 observations of the 138-row subset are used for coefficient computation but not for the SSE calculation. The estimation window of 52 weeks (20 August 1997 to 12 August 1998) was chosen as the period during which Italy's EMU entry was considered highly likely by market participants.

---

## Scaling Convention

All raw spread data is in **annual percentage points** (e.g., a value of 3.6 means 3.6% per annum).

In the code, the data is divided by 5200 = 52 x 100 to convert to **weekly decimal** form:

```
dthat = raw_spreads / 5200
```

Model outputs are multiplied back by 5200 for plotting in percentage points.

---

## Key Dates for Figure 2

| Date | Weeks to EMU | Quarters to EMU | Est. window index (0-based) |
|------|-------------|-----------------|----------------------------|
| 01.10.1997 | ~65 | ~5 | 6 |
| 31.12.1997 | ~52 | ~4 | 19 |
| 01.04.1998 | ~39 | ~3 | 32 |
| 01.07.1998 | ~26 | ~2 | 45 |

---

## References

- Lund, J. (1999). A model for studying the effect of EMU on European yield curves. *European Finance Review*, 2, 321--363.

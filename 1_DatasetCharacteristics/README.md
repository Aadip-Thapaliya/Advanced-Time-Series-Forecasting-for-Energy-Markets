# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source
- **Dataset Link:** "https://www.energy-charts.info/charts/power/chart.htm?c=DE&legendItems=0wm&interval=week&year=2025&source=public"
- **Dataset Owner/Contact:** It's a collection of public data. Collected by Fraunhofer Institute for Solar Energy Systems ISE. The sources are "https://www.energy-charts.info/sources.html?l=en&c=DE"

### Dataset Characteristics
- **Number of Observations:** Samples: 70176 and the intervall is 15minutes.
- **Number of Features:** for 23 features

### Target Variable/Label
- **Label Name:** "Day Ahead Auction (DE-LU)"
- **Label Type:** "Regression"
- **Label Description:** The label present the day ahead auction price for electricity in germany.  
- **Label Values:** The value can be negative or positive. typically between -125EUR/MWh and 400 EUR/MWh
- **Label Distribution:** There is no trend in the data but a high seasonality.

### Feature Description
[Provide a brief description of each feature or group of features in your dataset. If you have many features, group them logically and describe each group. Include information about data types, ranges, and what each feature represents.]

**All Power MW:**
- **Feature 1 :** Hydro pumped storage consumption 
- **Feature 2 :** Cross border electricity trading
- **Feature 3 :** Hydro Run-of-River
- **Feature 4 :** Biomass
- **Feature 5 :** Fossil brown coal / lignite
- **Feature 6 :** Fossil hard coal
- **Feature 7 :** Fossil oil
- **Feature 8 :** Fossil coal-derived gas
- **Feature 9 :** Fossil gas
- **Feature 10 :** Geothermal
- **Feature 11 :** Hydro water reservoir
- **Feature 12 :** Hydro pumped storage
- **Feature 13 :** Others
- **Feature 14 :** Waste
- **Feature 15 :** Wind offshore
- **Feature 16 :** Wind onshore
- **Feature 17 :** Solar
- **Feature 18 :** Load
- **Feature 19 :** Residual load
**Renewable share (%):**
- **Feature 20 :** Renewable share of load
- **Feature 21 :** Renewable share of generation
**Price (EUR/MWh):**
- **Feature 22 :** Day Ahead Auction (DE-LU)

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment

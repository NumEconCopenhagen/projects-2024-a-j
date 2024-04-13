# Data analysis project

Our project is titled **"Electricity: Prices and production from renewables in Eastern Denmark (2023)"** and explores the relationship between electricity prices and the production of electricity from renewable sources within the price area "DK2" - Eastern Denmark. To gather data for this, we utilize the API service publicly offered by the Danish TSO (Transmission System Operator) Energinet, known as Energidataservice. We gather data from two distinct datasets: "Elspot Prices" and "Production and Consumption Settlement", which provide information on prices, as well as the amounts of electricity produced and consumed, in hourly increments throughout the entire year of 2023.

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).

We apply the **following datasets**:

1. elspot_prices.csv (*called from Energidataservice API*) 
1. production_consumption.csv (*called from Energidataservice API*)

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install requests``

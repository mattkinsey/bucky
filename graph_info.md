# Spatial SEIR Models Overview
Running the spatial SEIR model is effectively a two-step process. First, a graph is generated with input data, which includes population data, case data, and mobility information. This graph is then used as input to the model. 

# Constructing the Graph
The graph is created using admin2-level data. If data can not be found at the admin2-level, admin2-level information can be extrapolated using admin2 population and national or state level data (this is expanded upon in the *Population Data* section).

The following data sources are used to create the graph:
* admin2-level shapefile
* admin2-level population data stratified by age
* Historical admin2-level case and death data
* Contact matrix information for the country
* Mobility data (or a proxy)

All data is placed into a single dataframe, joined by the admin2-level key (e.g., FIPS for United States), with the exception of mobility data (which is used to create edges, not nodes).

*NOTE*: This graph should be able to be constructed on the province or state level, but may require model changes. If this is the case, talk to Matt Kinsey.

## Graph-Level Attributes
Administrative information is placed on the top graph-level. For example:
```
'adm1_key' : 'STATEFIPS',
'adm2_key' : 'FIPS',
'adm1_to_str' : {1 : 'Alabama'},
'adm0_name': 'US',
'start_date' : '2020-09-25'
'adm1_to_str' is a dict with key-value pairs indicating the adm1 names for each adm1 value appear in the graph. 
```
Contact matrices are also on this level under the key `'contact_mats'`.

## Sample Node
The following is an example node on the graph for the United States. The rest of the documentation will describe what data is necessary to construct this node.

```python
(0,
 {'STATEFIPS': 1,
  'Confirmed': 1757.0,
  'Deaths': 25.0,
  'adm2_name': 'Autauga County',
  'N_age_init': array([3364., 3423., 3882., 3755., 3173., 3705., 3461., 3628., 3616.,
         3966., 3811., 3927., 3237., 2589., 2311., 3753.]),
  'Population': 55601.0,
  'IFR': array([6.75414158e-06, 1.24643105e-05, 2.26550214e-05, 4.05345945e-05,
         7.68277004e-05, 1.38382882e-04, 2.54273120e-04, 4.63844627e-04,
         8.51898589e-04, 1.55448599e-03, 2.87077658e-03, 5.20528393e-03,
         9.47735996e-03, 1.73603179e-02, 3.14646839e-02, 9.38331984e-02]),
  'case_hist': array([1207.64227528, 1234.9656055 , 1243.85366911, 1244.13444753,
         1255.27521116, 1268.95333353, 1270.38458817, 1288.05778954,
         1295.55174933, 1297.29129258, 1312.35308192, 1321.2898892 ,
         1323.10534634, 1354.97350342, 1358.88036484, 1362.43488575,
         1377.67551466, 1392.4338964 , 1406.70605635, 1446.3924143 ,
         1450.47616771, 1462.67851762, 1458.98710032, 1470.52271903,
         1481.12998684, 1501.75698721, 1508.06090303, 1513.9178672 ,
         1518.26245703, 1532.99858052, 1553.97101414, 1564.24619451,
         1579.10859377, 1590.56170754, 1597.77332362, 1616.97996262,
         1619.        , 1624.        , 1664.        , 1673.        ,
         1690.        , 1691.        , 1714.        , 1715.        ,
         1738.        , 1757.        ]),
  'death_hist': array([22.76748794, 22.80142062, 22.81307638, 22.79580414, 22.79344408,
         22.79578013, 22.81581338, 22.80532061, 22.7902682 , 22.79603286,
         22.79689139, 22.79601336, 22.79344923, 22.85912123, 22.90405033,
         22.91397178, 22.97898824, 23.02565004, 23.05597481, 23.09719551,
         23.13913548, 24.12323294, 24.17184064, 24.2852927 , 24.38579416,
         24.41284998, 24.41330133, 24.41175889, 24.40910247, 24.41419481,
         24.43286524, 24.47610337, 24.52580854, 24.5245916 , 24.53522989,
         24.54591406, 24.        , 24.        , 24.        , 24.        ,
         24.        , 24.        , 25.        , 25.        , 25.        ,
         25.        ]),
  'FIPS': 1001.0})
```

## Population Data
Population data should be at a admin2 level and stratified in 16 5-year age bins (if using Prem et al contact matrices): 

* 0-4 years
* 5-9 years
* 10-14 years
* 15-19 years
* 20-24 years
* 25-29 years
* 30-34 years
* 35-39 years
* 40-44 years
* 45-49 years
* 50-54 years
* 55-59 years
* 60-64 years
* 65-69 years
* 70-74 years
* 75+ years

If population data for an admin2 area is known (i.e. number of total people per admin2), but it is not age-stratified, this data can be extrapolated assuming age-stratified population data exists at some level. For example, assume a country has age-stratified data provided at the national-level. To get the admin2-level age data, the data is separated into the 16 bins (as a 1-dimensional array of length 16). These bins are then normalized by dividing by the sum. Then, the fraction of people living in the admin2 is calculated by dividing admin2 population by the total national population. For each district, this fraction is multiplied by the age vector to produce a admin2-level age vector. This vector is placed on the node under the key *N_age_init*.

The total population for a admin2 is placed on the node under the key *Population*.

## Case Data
Case data should be at the admin2-level and include cumulative data as of the start date of the simulation and historical data for the 45-day period preceding the start date:

* Confirmed: Cumulative number of cases per admin2 region as of the start date of the simulation
* Deaths: Cumulative number of deaths per admin2 region as of the start date of the simulation
* case_hist: Historical case data (i.e. "Confirmed") data
* death_hist : Historical death data (i.e. "Deaths") data

Historical data is the most important of these to have as it is used to initialize infections.

*Confirmed* and *Deaths* are placed on the node with their respective keys. Historical data is structured as numerical vectors on the node with the keys *case_hist*, *death_hist*. If historical data is not available for the requested date range, the array is filled with zeros. If there are gaps in the historical data, they are replaced using pandas' *bfill* function, which uses the next valid data point to fill the gap.

## Contact Matrices
Currently, contact matrix data is downloaded from [here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005697), which has contact matrices for 152 countries. If a country does not appear in this dataset, a country culturally close can be substituted (for example, Pakistan's contact rates were used for Afghanistan), or another dataset can be used. If another dataset is used, the contact matrix must be formatted such that it has the same shape as the number of age demographic bins (i.e. if there are 16 bins, the matrix must be of size 16 x 16).

## Mobility Data
Mobility data is used to construct the edges of the graph. Mobility data, or a proxy for it, is used to describe the contact rates between counties.

The baseline mobility data shows up as an edge attributed called *weight*. *R0_frac* is a factor that is multiplied with the baseline mobility value to model the effect of NPIs, etc., on mobility. For example, given baseline mobility data from February 2020, *R0_frac* would be computed by dividing recent mobility data values with the February 2020 baseline. *R0_frac* exists to provide a knob to tune during the simulation to model NPIs.

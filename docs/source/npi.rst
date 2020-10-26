================================
Non-pharmaceutical Interventions
================================

Non-pharmaceutical interventions (NPIs) are mitigations, apart from getting vaccinated and taking medicine, that people and communities can take to help slow the spread of communicable diseases. As a vaccine for COVID-19 has yet to be deployed, NPIs are among the best strategies for controlling the spread of the current COVID-19 virus.  The structure of the Bucky model allows for the incorporation of NPIs via the modification of a combination of the following : the basic reproduction number, local contact matrices, and inter-regional mobility matrices. 

For each country an initial list of NPIs was obtained from the ACAPS `COVID-19 Government Measures Dataset <https://data.humdata.org/dataset/acaps-covid19-government-measures-dataset>`_ . This dataset is complemented with additional qualitative information from in-country stakeholders. The estimated compliance level are tailored to specific countries.

Implementation
-------------------------------------------------
    
NPIs are categorized and implemented in Bucky based on their classification into three categories:

Contact-Matrix Based NPIs
*************************

These NPIs are those that effect only certain age groups within the total population.  These NPI effect the ratios relating the components of the contact matrices. The NPI that fall under this category are:

- School Closure
- Shielding Elderly

Mobility Based NPI
******************

This classification is for those NPI that lead to changes in mobility/movement between administrative districts (as opposed to movement within an administrative district).  The NPI that fall under this category are :

- Closing of borders, ports, and/or international flights
- Restricting inter-regional movement

Reproduction Number Based NPI
*****************************

This classification is for those NPI that have an effect on the overall scaling of transmissibility.  It encompasses both intra-regional measures to reduce transmission as well as national level initiatives designed to reduce transmission throughout the country. The NPI that fall under this category are :

- Social distancing
- Face mask wearing
- Installation of hand washing stations
- Reduction of size of public gatherings
- Closing businesses
- Partial Lockdown
- Awareness campaigns (e.g., vaccination programs)



A summary of the NPIs that are currently implemented in Bucky are given in the table below.  This table includes the classification, effects, and sources that are currently being used to approximate the effects of various NPI.

With the current implementation, we have the ability to distinguish between the effects of NPI within the categories mentioned above. For the case in which multiple NPI within category III are implemented, we have implemented a value-added approach to calculating their effectiveness in reducing the basic reproduction number. In this case, we calculate the reduction in :math:`R_0`  based on the number of NPIs in place.  If one NPI is in place, :math:`R_0` is reduced by 40\%. If two NPI are in place, :math:`R_0` is reduced by 60\%.  If three or more NPI are in place, then :math:`R_0` is reduced by 70\%.  

=============================  ===========================================================================================  ================================================  =============== 
NPI Classification             Effect in Model                                                                              Mean Reduction (SD)                               Source
=============================  ===========================================================================================  ================================================  =============== 
Contact-based: School closure  Reduce contact between school aged groups and increase the contacts in the home environment  ~44% reduction in overall community transmission  :cite:`Wells7504`
Mobility-based                 Reduction in mobility between regions                                                        60% (10)                                          :cite:`Cowling2020` :cite:`Wells7504`
Reproduction number-based      60-8% reduction in overall community transmission                                            72.5% (6.25)                                      :cite:`jarvis2020quantifying` :cite:`joseph2020assessing`
=============================  ===========================================================================================  ================================================  =============== 

Creating the NPI file
---------------------
The NPI file has to consist of the columns `admin2, date, home, school, work, other_locations, mobility_reduction,r0_reduction`.
Each row contains an unique `admin2`-`date` combination. The other columns indicate the effect of the current NPIs on the given column.
A value of 1 indicates the NPIs having no effect on the transmission for the given column name. A value smaller than 1 indicates that the NPI's lower the transmission in this domain, while values larger than 1 indicate an increased transmission.

The columns `home, work, school, other_locations,` correspond with the columns in the contact matrix. The contact matrix is multiplied with the values of these columns in the NPI file.
The value in `mobility_reduction` influences the mobility between regions, each weight in the mobility matrix that has as start or endpoint the given admin2, will be multiplied by that `mobility_reduction` value. Lastly, the effective reproduction rate is multiplied with the `r0_reduction` to come to the new effective reproduction rate given the NPIs.

An example file is given in *included_data/npi/AFG_NPIs_20201014.csv*. The `OCHA model parameterization repository <https://github.com/OCHA-DAP/pa-COVID-model-parameterization>`_ implements a method to convert written NPI measures to the required Bucky format.  This script also generates an `elderly_shielding` column, but this NPI is currently not implemented in the model and thus not required.

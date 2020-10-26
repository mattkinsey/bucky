======================
Model Input and Output
======================

Input
=====
The Bucky model uses two main sources of input: the input graph and CDC-recommended parameters.  

Input Graph
-----------
The input graph contains data regarding demographics, geographic information, and historical case information. For details, see :doc:`Graph Information <graph_info>`.

CDC-Recommended Parameters
--------------------------
The Centers for Disease Control and Prevention (CDC) has published pandemic planning scenarios :cite:`centers2020covid` which contain recommended parameters describing biological and epidemiological parameters. Of these five planning scenarios, Bucky utilizes scenario 5, which contains the CDC’s current best estimates for disease severity and transmission. These parameters are described in detail, based on information available from the CDC, and summarized in the table below. CDC-recommended parameters are controlled by parameter files located in the ``par`` directory.

===================================================  =====================  ==============
Parameter Description                                Bucky Variable Name    Value (Interquartile Range)
===================================================  =====================  ==============
Mean generation interval                             T_g                    7.5 (5.5, 8.5) 
Mean serial interval                                 T_s                    6 (5, 7)
Fraction of infections that are asymptomatic         asym_frac              0.4
Relative infectiousness of asymptomatic individuals  rel_inf_asym           0.75 
Percentage of transmission prior to symptom onset    frac_trans_before_sym  0.5
Case fatality ratio                                  F                      - 0-49 years: 0.0005
                                                                            - 50-64 years: 0.002
                                                                            - 65+ years: 0.013                
Case hospitalization ratio                           H                      - 0-49 years: 0.017
                                                                            - 50-64 years: 0.045
                                                                            - 65+ years: 0.074
TIme from symptom onset to hospitalization           I_TO_H_TIME            - 0-49 years: 6 days
                                                                            - 50-64 years: 6 days
                                                                            - 65+ years: 4 days 
Duration of hospitalization                          H_TIME                 - 0-49 years: 4.9 days
                                                                            - 50-64 years: 7.6 days
                                                                            - 65+ years: 8.1 days 
Time between death and reporting                     D_REPORT_TIME          - 0-49 years: 7.1 days
                                                                            - 50-64 years: 7.2 days
                                                                            - 65+ years: 6.6 
===================================================  =====================  ==============

Disease Transmission
********************

The following parameters describe the transmissibility of the virus. The **percentage of infections that are asymptomatic**, ``asym_frac``, refers to the percentage of infections that will never develop symptoms. This is a difficult parameter to estimate due to logistical complications (individuals would need to be tested to ensure they remain asymptomatic while infectious) and because the level of asymptomatic infections varies by age. The best estimate for this parameter is the midpoint between the lower bound of :cite:`byambasuren2020estimating`, the upper bound of :cite:`poletti2020probability`, which corresponds to the estimates from :cite:`oran2020prevalence`. 

The **relative infectiousness of asymptomatic individuals** compared to symptomatic individuals ``rel_inf_asym`` is calculated using upper and lower bounds on the difference in viral dynamics between asymptomatic and symptomatic cases. The lower bound is derived from data indicating that more severe cases have higher viral loads :cite:`liu2020viral` and a study that indicates symptomatic cases shed for longer and have higher viral loads than asymptomatic cases :cite:`noh2020asymptomatic`. Other studies indicate that both symptomatic and asymptomatic cases have similar duration and viral shedding :cite:`lee2020clinical`, which is used as the upper bound. 

The final parameter relating to disease transmission is the **fraction of transmission prior to symptom onset** ``frac_trans_before_sym`` which corresponds to the percentage of new cases that were caused by transmission from an individual before they become symptomatic. The lower bound is derived from :cite:`he2020temporal`, with the upper bound derived from :cite:`casey2020estimating`.

Disease Characteristics and Severity
************************************

The mean serial interval, ``Ts``, is the time in days from exposure to onset of symptoms and is taken from :cite:`mcaloon2020incubation`. The mean generation interval, ``Tg``, is the period of time (in days) between symptom onset for one individual and symptom onset for a person they have infected. This value is from :cite:`he2020temporal`. 

The case fatality ratio (**CFR**) is the number of individuals who will die of the disease; the case hospitalization-severity ratio (**CHR**) corresponds to the number of cases that are severe and necessitate hospitalization. Within the context of the United States, this ratio corresponds to the individuals admitted to a hospital.  In a context where access to medical care is limited, this ratio corresponds to the ratio of individuals who exhibit severe disease symptoms.

Hospital-related parameters are derived using data from COVID-Net :cite:`covid-net` and the CDC's Data Collation and Integration for Public Health Event Response (DCIPHER). All data is taken from the period between March 1, 2020 to July 15, 2020 unless otherwise noted. The time it takes from symptom onset to hospitalization in days is denoted by ``I_to_H_time``. The number of days an individual will be hospitalized is ``H_TIME``. Finally, the number of days between death and reporting is ``D_REPORT_TIME``.

Output
======
The Bucky model generates one file per Monte Carlo run. This data is post-processed to combine data across all dates and simulations. It can then be aggregated at desired geographic levels. A separate file is created for each requested administrative levevel, with each row indexed by data, admin ID, and quantile. The columns of this output file are described in the tables below.

==========  ===========
Index name  Description
==========  ===========
adm*        The adm ID corresponding to the geographic level (i.e. adm2 ID)
date        The date
quantile    Quantile value     
==========  ===========


==================================  ===========
Column name                         Description
==================================  ===========
case_reporting_rate                 Case reporting rate
active_asymptomatic_cases           Current number of actively infectious but asymptomatic cases
cumulative_cases                    Cumulative number of cumulative cases (including unreported)
cumulative_deaths                   Cumulative number of deaths
cumulative_deaths_per_100k          Cumulative number of deaths per 100,000 people
cumulative_reported_cases           Cumulative number of reported cases
cumulative_reported_cases_per_100k  Number of reported cumulative cases per 100,000 people
current_hospitalizations            Current number of hospitalizations
current_hospitalizations_per_100k   Number of current hospitalizations per 100,000 people
current_icu_usage                   ICU bed usage
current_vent_usage                  Current ventilator usage
total_population                    Population
daily_cases                         Number of daily new cases (including unreported)
daily_deaths                        Number of daily new deaths
daily_hospitalizations              Number of daily new hospitalizations
daily_reported_cases                Number of reported daily new cases
doubling_t                          Local doubling time as estimated from the historical data
R_eff                               Local effective reproductive number
==================================  ===========

======================
Model Input and Output
======================

Input
=====
The Bucky model uses two main sources of input: the input graph and CDC-recommended parameters. The input graph contains data regarding demographics, geographic information, and historical case information.

CDC-Recommended Parameters
--------------------------
The Centers for Disease Control and Prevention (CDC) has published pandemic planning scenarios [7] which contain recommended parameters describing biological and epidemiological parameters. Of these five planning scenarios, OCHA-Bucky utilizes scenario 5, which contains the CDC’s current best estimates for disease severity and transmission. These parameters are described in detail, based on information available from the CDC, and summarized in Table 2.

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


Output
======
The Bucky model generates one file per Monte Carlo run. This data is post-processed to combine data across all dates and simulations. It can then be aggregated at desired geographic levels. A separate file is created for each requested administrative levevel, with each row indexed by data, admin ID, and quantile. The columns of this output file are described in the tables below.

==========  ===========
Index name  Description
==========  ===========
adm*        The adm ID corresponding to the geographic level (i.e. adm2 ID)
date        The date
q           Quantile value     
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

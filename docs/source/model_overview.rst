Model Description
=================

The JHUAPL-Bucky model is a COVID-19 metapopulation compartment model initially designed to estimate medium-term (on the order of weeks) case incidence and healthcare usage at the second administrative (admin-2, ADM2) level (counties in the United States; cities or districts in various countries).
These ADM2 regions are all coupled using mobility information to approximate the both inter- and intra-regional contacts between the members of the populations.
Using the historical case and death data, local demographic data (see :doc:`Graph Information <graph_info>`), and a set of :doc:`parameters <input_output>` derived from empirical studies, the model infers a number of localized features (see table below) that are related to spread of COVID-19.
Projecting forward in time, Bucky then utilizes an age stratified compartment model to not only estimate the case load but additionally provide outputs relating to the healthcare burden of each locality.

These time forecasts are performed a large number of times (Monte Carlo experiments), with each individual simulation using minor modifications to the input parameters at random, scaled to the uncertainty of the estimates. The resulting collection of simulations is then used to obtain probabilistic estimates for all output variables.



Model Overview
--------------


At its base, the Bucky model is a spatially distributed SEIR model. SEIR models are a class of deterministic models used to model infectious diseases that are spread by person-to-person transmission in a population. The simplest versions of such models are systems of ordinary differential equations and are analysed mathematically :cite:`Hethcote1989`. 

Within the context of an SEIR model, disease dynamics are modeled over time by moving the population through a series of compartments (otherwise known as "bins" or "states").  Those states are as follows:

    -  susceptible (S): the fraction of the population that could be potentially subjected to the infection;
    -  exposed (E): the fraction of the population that has been infected but does not show symptoms yet;
    -  infectious (I): the fraction of the population that is infective after the latent period;
    -  recovered (R): The fraction of the population that have been infected and recovered from the infection.


The total population is represented by the sum of the compartments.  Basic assumptions of this type of model include: 

    -  Once the model is initialized, no individuals are added to the susceptible group.  It follows that births and natural deaths are unaccounted for, migration in/out of the region is frozen for the duration of a simulation, and none of the population has been vaccinated or is immune to the pathogen;
    -  The population within each strata is uniform and each pair of individuals within the strata are equally likely to interact;
    -  The probability of interaction between individuals in the population is not rare;
    -  Once infected, an individual cannot be reinfected with the virus. 


.. tikz:: Model
    :libs: shapes.geometric,backgrounds, arrows, positioning

    \begin{scope}[node distance=3.5cm and 3cm]
    \node (S) [square] {$\text{S}_{ij}$};
    \node (E) [square, right=1cm of S] {$\text{E}_{ij}$};
    \node (IH) [square, below right=2cm and 4cm of E] {$\text{I}_{ij}^{\text{hosp}}$};
    \node (IM) [square, right =4cm of E] {$\text{I}_{ij}^{\text{mild}}$};
    \node (IA) [square,  above right=2cm and 4cm of E] {$\text{I}_{ij}^{\text{asym}}$};
    \node (R) [square,  right= 5cm of IM] {$\text{R}_{ij}$};
    \node (RH) [square, below right= .3cm and 2cm of IM] {$\text{R}_{ij}^{\text{hosp}}$};
    \node (D) [square, right =5cm of IH] {$\text{D}_{ij}$};
    \end{scope}
    \draw[arrow] (S) -- (E) node[midway,above] {$\beta_{ij}$};
    \draw[arrow] (E) -- (IH) node[midway,sloped, above]{$(1-\alpha)\eta_{i} \sigma$};
    \draw[arrow] (E) -- (IM) node[midway,above] {$(1-\alpha) (1-\eta_{i}) \sigma$};
    \draw[arrow] (E) -- (IA) node[midway, sloped, above] {$\alpha(1-\eta_{i}) \sigma$};
    \draw[arrow] (IM) -- (R) node[midway, above] {$\gamma$};
    \draw[arrow] (IA) -- (R) node[midway, above] {$\gamma$};
    \draw[arrow] (IH) -- (RH) node[midway, sloped, above] {$(1-\phi_i)\gamma$};
    \draw[arrow] (RH) -- (R) node[near start, above] {$\tau_i$};
    \draw[arrow] (RH) -- (D) node[midway, sloped,above] {$\phi_i \gamma$};


.. note:: The compartments :math:`\text{E}`, :math:`\text{I}^{\text{asym}}`, :math:`\text{I}^{\text{mild}}`, :math:`\text{I}^{\text{hosp}}` and :math:`\text{R}^{\text{hosp}}` are gamma-distributed with shape parameters specified in the configuration file.



The Bucky model consists of a collection of coupled and stratified SEIR models. Since COVID-19 exhibits heavily age dependent properties, wherein a majority of severe cases are in older individuals, SEIR models are stratified via the age demographic structure of a geographic region in order to get accurate estimates of case severity and deaths.  Additionally, to model the spatial dynamics of COVID spread, we consider a set of SEIR sub-models at the smallest geographic level for which we have appropriate data.

The basic structure of the model is displayed in the diagram above. Age is denoted by index *i*, and geographic regions are denoted by index *j*. Within each strata, Bucky models the susceptible and exposed populations, followed by one of three possible infected states: asymptomatic (:math:`\text{I}^{\text{asym}}`), mild (:math:`\text{I}^{\text{mild}}`), and severe (:math:`\text{I}^{\text{hosp}}`).  Members of the population who are either asymptomatic or exhibit mild symptoms recover from the virus at a rate :math:`\gamma`.  Those who exhibit severe symptoms and are in need of healthcare support will either recover after a period of illness at rate :math:`1/\tau_i` or expire as a result of the virus at rate :math:`\phi_i \gamma`. 

A critical component of the Bucky model is the parameterization of the model.  A number of parameters must be derived and/or estimated from their original data sources.  These include, but are not limited to those listed in tables below as well as local estimates of local case doubling time, case reporting rate, case fatality rate, and the case hospitalization rate.  Further details of these quantities as well as how they are estimated are given in the :doc:`Model Input and Ouput section <input_output>`. All parameter estimation for the model includes the basic assumption that, once estimated and initialized, these parameters remain constant during the simulation period. 

Coupling individual age and geographically stratified sub-models occurs across a number of dimensions including disease state. Sub-models are coupled together using both the spatial mobility matrix and age-based contact matrices. Modeling of the overall interaction rates between geographic locations and age groups is an important component in accurately modeling non-pharmaceutical Interventions (NPIs).  Bucky accounts for the implementation of NPIs (e.g. school closures, border closures, face mask wearing) via modifying either the social contact matrices or the basic reproductive number, :math:`R_0`. For further details, see :doc:`Non-pharmaceutical Interventions <npi>`.

All together, these components contribute to a model that is adaptable to a number of contexts. Bucky is calibrated to the uncertainties in both the case data and the disease parameters, leading to a model that is robust to both the quality and resolution of available input data.

========================  ===========
Variable                  Description
========================  ===========
:math:`S_{ij}`            Proportion of individuals who are susceptible to the virus
:math:`E_{ij}`            Proportion of individuals who have been exposed to the virus
:math:`I_{ij}^{hosp}`     Proportion of individuals that are exhibiting severe disease symptoms and are in need of hospitalization
:math:`I_{ij}^{mild}`     Proportion of individuals that are exhibiting mold disease symptoms
:math:`I_{ij}^{asymp}`    Proportion of individuals who are infected but asymptomatic
:math:`R_{ij}`            Proportion of individuals who have recovered from the virus and are no longer capable of infecting other individuals
:math:`R_{ij}^{hosp}`     Proportion of individuals who have recovered from the virus after a period of time in a hospital
:math:`D_{ij}`            Proportion of individuals who have succumbed as a direct result of the virus
========================  ===========


========================  =======
Parameter                 Description
========================  =======
:math:`\beta_{ij}`        Force of infection on a member of age group *i* in location *j*
:math:`\frac{1}{\sigma}`  Viral latent period
:math:`\alpha`            Rate of infections that are asymptomatic
:math:`\eta_i`            Fraction of cases necessitating hospitalization 
:math:`\phi_i`            Case fatality rate for age group *i*
:math:`\frac{1}{\gamma}`  Infectious period
:math:`\tau_i`            Recovery period from severe infection for age group *i*
========================  =======
# OCHA-Bucky Model
![alt text](https://github.com/OCHA-DAP/pa-ocha-bucky/blob/ocha/logo.png?raw=true)


[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://ocha-bucky.readthedocs.io/en/latest/getting_started.html)
![Lint Passing (master)](https://github.com/mattkinsey/bucky/workflows/test-pre-commit-hooks/badge.svg?branch=master)

The documentation of the model, including how to get started, can be accessed **[here](https://ocha-bucky.readthedocs.io/en/latest/)** 

If you have any questions or feedback, contact us at centrehumdata@un.org.

## About the model
The OCHA-Bucky model is a spatial SEIR model for simulating COVID-19 at the subnational level. 
The model has been developed as a partnership between UN OCHA and the Johns Hopkins University Applied Physics Laboratory (JHU/APL). 
The OCHA-Bucky model includes a series of adjustments to a novel COVID-19 model ([JHUAPL-Bucky](https://github.com/mattkinsey/bucky)) that incorporates different vulnerability factors 
to provide insights on the scale of the crisis in priority countries at national and regional levels, 
how different response interventions are expected to impact the epidemic curve, 
and the duration of the crisis in specific locations. 
OCHA-Bucky stratifies COVID-19 dynamics by age and population vulnerability. 
Input to the model consists of geographically distributed COVID cases and deaths. 
Model output consists of future projections of these same quantities, as well as severe cases (defined as a proportion of total cases). 
The model considers both inter-regional mobility of the population and time-varying non-pharmaceutical interventions (NPIs). 

OCHA-Bucky has already been used to provide bi-weekly [projection reports](https://drive.google.com/drive/u/1/folders/16FR8owccpfIm-tspdAa4YTEwPoZKHtvI) to six OCHA country offices: Afghanistan, the Democratic Republic of Congo, Iraq, Somalia, Sudan and South Sudan. To this end a separate [OCHA model parametrization repository](https://github.com/OCHA-DAP/pa-COVID-model-parameterization) exists for creating the desired inputs for OCHA-Bucky.

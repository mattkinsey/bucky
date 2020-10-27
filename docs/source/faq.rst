Frequently Asked Questions
==========================

Is it possible to change the default parameters?
------------------------------------------------
The model uses a set of default parameters which are defined in `par/scenario_5.yml`. You can change the parameters in this file, or add your own file as long as it follows the same format. This file can be added as input to `model.py` with the `par_file` argument.

How does Bucky handle missing subnational data?
-----------------------------------------------
The model assumes all missing values to be zeroes. The structure of the model is such that it is assumed all adjustments to the input data are done in the process of making the graph. This keeps the model itself a clean numerical integrator which throws an error if the data is incorrect instead of trying to fix it. Thus, if it is desirable to apply other techniques for missing values, this should be done while creating the graph.

I see different branches in the repository. What are the differences?
---------------------------------------------------------------------
The main branches in use are the `master` and `ocha` branch. The `master` branch is optimised for usage in the US while the `ocha` branch is optimized for usage in the humanitarian context. The `ocha` includes all the additional features specific to humanitarian context done by the Centre and Johns Hopkins University Applied Physics Laboratory

Are there situations in which the model output might not be reliable/the model should be used with caution?
------------------------------------------------------------------------------------------------------------
This is difficult to say with specificity. However, it is advised to always test the projections against the input data to detect odd trends. This can be for example done by doing a run with a start date of ~1 month ago and comparing the results with the historical data. Plotting the historical data is implemented in `viz/plot.py`. There will be discrepancies in places if there were wildly changing dynamics but there should be reasonable agreement between the projections and historical data. If not, the input data and parameters should be checked.

How can I change the number of days used to calculate the doubling time from historical data?
---------------------------------------------------------------------------------------------
This is one of the parameters defined in `par/scenario_5.yml` and thus can be changed here. Its default value is 7 days but for countries with limited reporting it might be needed to increase this number. A too low doubling time can also be a cause of error if the cumulative cases does not change during the doubling time window. Adding `-vv` to the model call will output inf values for the doubling time if this is the cause of error.

At what admin level are the simulations run and can I change this?
------------------------------------------------------------------
The simulations are being run at admin 2 level. This level is hardcoded in many places in the model so, in short, it is not easy to change the level. The `postprocess` script does allow aggregation to admin1 and admin0 level. If you want to run the model on admin1 or admin0 level a workaround is to just add one admin2 region per admin1 to the Graph file.

How is the effective reproductive number estimated?
---------------------------------------------------
The effective reproductive number (R_eff) is estimated from historical data. The OCHA-Bucky model uses historical national data from WHO for the computation.

What is the start date of the simulation and can I change this?
---------------------------------------------------------------
The start date of the simulation is the start date as specified in the input graph as a graph-level property. This start date is assumed to equal the last date of the historical data in the graph.

Can the simulation start at a date later than the last subnational data point?
------------------------------------------------------------------------------
No, the model initializes based on the last data point and thus this has to equal the start date.

What is the maximum number of historical days Bucky uses?
----------------------------------------------------------
This depends on the parameters of the doubling time and case reporting rate estimates. The minimum is 45 days, but it does not harm to have a longer history in the graph file.
#!/bin/bash

set -xe

N_MC=1000
N_DAYS=60
N_PROC=8 # lower this is you run out of mem during postprocess
NPI_DIR='./data/npi'
GRAPH_DIR='./un_graphs'

export CUDA_VISIBLE_DEVICES=0
# Uncomment the following line if you have cub and cutensor installed (it will speed up some things in cupy)
#export CUPY_ACCELERATORS=cub,cutensor

mkdir -p $NPI_DIR

for c in IRQ SDN SOM AFG COD SSD;
do
   # get most up to date npi file
   pushd data/npi
   rm -f ${c}_NPIs.csv
   curl -kL https://raw.githubusercontent.com/OCHA-DAP/pa-COVID-model-parameterization/master/Outputs/${c}/NPIs/${c}_NPIs.csv --output ${c}_NPIs.csv
   popd

   # get historical data off the graph
   ./bmodel util.graph2histcsv $GRAPH_DIR/${c}_graph.p $GRAPH_DIR/hist_${c}.csv

   #run model normally
   ./bmodel model -d $N_DAYS -n $N_MC -opt -g $GRAPH_DIR/${c}_graph.p --npi_file $NPI_DIR/${c}_NPIs.csv

   ./bmodel postprocess -n $N_PROC -l adm0 adm1 -g $GRAPH_DIR/${c}_graph.p --prefix ${c}_npi

   ./bmodel viz.plot -l adm0 adm1 -g $GRAPH_DIR/${c}_graph.p

   ./bmodel viz.plot -l adm0 adm1 --plot_columns current_hospitalizations daily_hospitalizations -g $GRAPH_DIR/${c}_graph.p

   #run model with --disable-npi
   ./bmodel model -d $N_DAYS -n $N_MC -opt -g $GRAPH_DIR/${c}_graph.p --npi_file $NPI_DIR/${c}_NPIs.csv --disable-npi

   ./bmodel postprocess -n $N_PROC -l adm0 adm1 -g $GRAPH_DIR/${c}_graph.p --prefix ${c}_no_npi

   ./bmodel viz.plot -l adm0 adm1 -g $GRAPH_DIR/${c}_graph.p

   ./bmodel viz.plot -l adm0 adm1 --plot_columns current_hospitalizations daily_hospitalizations -g $GRAPH_DIR/${c}_graph.p

done

#!/bin/bash

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:${s}[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

base_dir=$(pwd)

eval $(parse_yaml config.yml)

mkdir -p $data_dir
cd $data_dir

# CSSE case data
if [ ! -d "cases/COVID-19" ]; then
	mkdir -p cases && pushd cases > /dev/null
        echo Cloning CSSE repo
        git -c http.sslVerify=false clone https://github.com/CSSEGISandData/COVID-19.git
	popd > /dev/null
fi

# Descartes mobility data
if [ ! -d "mobility/DL-COVID-19" ]; then
	mkdir -p mobility && pushd mobility > /dev/null
        echo Cloning Descartes Labs mobility data
        git -c http.sslVerify=false clone https://github.com/descarteslabs/DL-COVID-19.git
	popd > /dev/null
fi

# COVIDExposureIndices mobility data
if [ ! -d "mobility/COVIDExposureIndices" ]; then
        mkdir -p mobility && pushd mobility > /dev/null
        echo Cloning COVIDExposureIndices mobility data
        git -c http.sslVerify=false clone https://github.com/COVIDExposureIndices/COVIDExposureIndices.git
        popd > /dev/null
fi

# US TL shapefiles
if [ ! -f "shapefiles/tl_2019_us_county.shp" ]; then
	mkdir -p shapefiles && pushd shapefiles > /dev/null
        curl -kL https://www2.census.gov/geo/tiger/TIGER2019/COUNTY/tl_2019_us_county.zip --output tl_2019_us_county.zip
	unzip -o tl_2019_us_county.zip
	#rm tl_2019_us_county.zip
	popd > /dev/null
fi

if [ ! -f "shapefiles/tl_2019_us_state.shp" ]; then
        mkdir -p shapefiles && pushd shapefiles > /dev/null
        curl -kL https://www2.census.gov/geo/tiger/TIGER2019/STATE/tl_2019_us_state.zip --output tl_2019_us_state.zip
        unzip -o tl_2019_us_state.zip
        #rm tl_2019_us_state.zip
        popd > /dev/null
fi

# US Census bridged-race population estimates (age stratified)
#https://www.cdc.gov/nchs/nvss/bridged_race/Documentation-Bridged-PostcenV2018.pdf
if [ ! -f "population/US_pop.csv" ]; then
        mkdir -p population && pushd population > /dev/null
        [ ! -f "pcen_v2018_y1018.txt.zip" ] && curl -kL https://www.cdc.gov/nchs/nvss/bridged_race/pcen_v2018_y1018.txt.zip --output pcen_v2018_y1018.txt.zip
        [ ! -f "pcen_v2018_y1018.csv" ] && unzip -p pcen_v2018_y1018.txt.zip pcen_v2018_y1018.txt/pcen_v2018_y1018.txt |
		 cut -c 5-11,86-93 | sed "s/./&,/7;s/./&,/5" > pcen_v2018_y1018.csv
	[ ! -f "US_pop.csv" ] && PYTHONPATH=$base_dir python -c "from bucky.util import bin_age_csv; bin_age_csv('pcen_v2018_y1018.csv','US_pop.csv')" 
        popd > /dev/null
fi

# Copy included data to data_dir
cp -nR $base_dir/included_data/* .

# Contact matrices
if [ ! -d "contact_matrices_152_countries" ]; then
        curl -kL https://doi.org/10.1371/journal.pcbi.1005697.s002 --output journal.pcbi.1005697.s002.zip
        unzip journal.pcbi.1005697.s002.zip       
fi

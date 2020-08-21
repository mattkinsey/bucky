Getting Started
===============

git clone https://gitlab.com/kinsemc/bucky.git
conda env create --file enviroment[_gpu].yml
conda activate bucky[_gpu]
vim config.yml
./get_US_data.sh
python -m bucky.make_input_graph (defaults to 2wk ago)
python -m bucky.model -n 100
python -m bucky.postprocess
python -m bucky.viz.plot
TODO add DMV lookup example

Model Flow
=================

.. tikz:: Model
    :libs: shapes.geometric,backgrounds, arrows, positioning, calc


    \tikzstyle{square} = [rectangle, rounded corners, minimum width=1.3cm, minimum height=1cm,text centered, draw=green, fill=green!30, line width=.5mm, align=center]
    \tikzstyle{squareb} = [rectangle, rounded corners, minimum width=1.3cm, minimum height=1cm,text centered, draw=blue, fill=blue!30, line width=.5mm, align=center]

    \tikzstyle{arrow} = [thick, ->]

    \begin{scope}[node distance=3.5cm and 3cm]
    \node (preprocess) [square] {Preprocess Historical Data/ \\ Anomoly Detection/Removal};
    \node (ingraph) [square, above=2cm of preprocess] {Standardized NetworkX Graph \\ (Country Agnostic)};
    \node (hist_data) [squareb, above right=2cm and .5cm of ingraph] {Historical Data \\ (CSSE, HHS, SafeGraph)};
    \node (demo_data) [squareb, above left=2cm and .5cm of ingraph] {Demographic Data \\ (Census)};

    \node (init) [square, right=3cm of preprocess] {Reset/ \\ Initialize Model State};
    \node (int) [square, below right=2cm and 1cm of init] {Integrate Model};
    \node (roll_params) [square, above=5cm of int] {Randomize Epi Parameters};
    \node (priors) [squareb, above=2cm of roll_params] {Epi Parameter Priors (CDC) / \\ Model Parameters};
    \node (post) [square, above right=2cm and 1cm of int] {Postprocess State \\ to Output Variables};
    \node (center) [above right =2.2cm and .1cm of int] {};
    \node (repeat) [above=2cm of int, align=center] {Repeat \\ N times};
    \node (quant) [square, right=3cm of post] {Compute Quantiles};
    \node (csv) [square, below left=2cm and 1cm of quant] {Convert to/Upload CSV};
    \node (valid) [square, above=2cm of quant] {Validation};
    \node (ensb) [square, below right=2cm and .5cm of quant] {Submit to CDC Ensemble};
    %\node (IH) [square, below right=2cm and 4cm of E] {$\text{I}_{ij}^{\text{hosp}}$};
    %\node (IM) [square, right =4cm of E] {$\text{I}_{ij}^{\text{mild}}$};
    %\node (IA) [square,  above right=2cm and 4cm of E] {$\text{I}_{ij}^{\text{asym}}$};
    %\node (R) [square,  right= 5cm of IM] {$\text{R}_{ij}$};
    %\node (RH) [square, below right= .3cm and 2cm of IM] {$\text{R}_{ij}^{\text{hosp}}$};
    %\node (D) [square, right =5cm of IH] {$\text{D}_{ij}$};
    \end{scope}
    \draw[arrow] (hist_data) -- (ingraph) [dashed] node[midway, left] {};
    \draw[arrow] (demo_data) -- (ingraph) [dashed] node[midway, left] {};
    \draw[arrow] (ingraph) -- (preprocess) node[midway, right] {GPU};
    \draw[arrow] (preprocess) -- (init) node[midway, above] {};
    \draw[arrow] (priors) -- (roll_params) node[midway, right] {GPU};
    \draw[arrow] (roll_params) -- (init) node[midway, above] {};
    \draw[arrow] (init) -- (int) node[midway, above] {};
    \draw[arrow] (int) -- (post) node[midway, above] {};
    \draw[->,>=latex'] (center) arc[radius=1.5cm,start angle=0,delta angle=270, line width=1cm];
    \draw[arrow] (post) -- (roll_params) node[midway, above] {};
    \draw[arrow] (post) -- (quant) node[midway, above] {Disk};
    \draw[arrow] (quant) -- (csv) [dashed] node[midway, above] {};
    \draw[arrow] (quant) -- (valid) [dashed] node[midway, above] {};
    \draw[arrow] (quant) -- (ensb) [dashed] node[midway, above] {};
    \draw[arrow] (valid) -- (priors) [dashed] node[midway, above] {};


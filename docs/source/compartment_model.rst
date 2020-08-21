Compartment Model Spec
======================

Diagram
-------

.. tikz:: Model
   :libs: shapes.geometric,backgrounds, arrows, positioning

   \begin{scope}[node distance=3.5cm and 3cm]
   \node (S) [square] {$S_{ij}$};
   \node (E) [square, below of=S] {$E_{ij}$};
   \node (TF) [square, below left of=E] {if $\Sigma_i I_{ij}^H < B_{j}$};
   \node (IH) [square, below right of=TF] {$I_{ij}^{hosp}$};
   \node (IS) [square, left of=IH] {$I_{ij}^{severe}$};
   \node (IM) [square, right of=IH] {$I_{ij}^{mild}$};
   \node (IA) [square, right of=IM] {$I_{ij}^{asym}$};
   \node (D) [square, below left of=IH] {$D_{ij}$};
   \node (RH) [square, right of=D] {$R_{ij}^{hosp}$};
   \node (R) [square,  below right of=RH] {$R_{ij}$};
   \node (dummy) [below=3.5cm of IS] {};
   \end{scope}
   \draw[arrow] (S) -- (E) node[midway,left] {$\beta_{ij}$};
   \draw[arrow] (E) -- (TF) node[midway,left]{$(1-\asym)H_i \sigma$};
   \draw[arrow] (E) -- (IM) node[midway, left] {$(1-\asym) \\ (1-H_i) \sigma$};
   \draw[arrow] (E) -- (IA) node[midway, right] {$\asym(1-H_i) \sigma$};
   \draw[arrow] (TF) -- (IH) node[midway, right] {True};
   \draw[arrow] (TF) -- (IS) node[midway, left] {False};
   \draw[arrow] (IS) -- (D) node[midway, right]{$\nu f_i \gamma$};
   \draw[arrow] (IH) -- (D) node[midway, right]{$f_i \gamma$};
   \draw[arrow] (IM) -- (R) node[midway, right] {$\gamma$};
   \draw[arrow] (IA) -- (R) node[midway, right] {$\gamma$};
   \draw[arrow] (IH) -- (RH) node[midway, right] {$(1-f_i)\gamma$};
   \draw[arrow] (RH) -- (R) node[midway, right] {$\rho$};
   \draw[->, >=latex', bend right=75, thick] (IS) to (R);
   \draw[arrow] (RH) -- (D) node[midway, below] {$f_i \gamma$};

Evolution Equations
-------------------

.. math::
   :nowrap:

   Where the compartments $E$, $I$ and $R^H$ are gamma-distributed with $k=2$
   
   Force of infection for age group $i$ at location $j$ is
   \begin{align*}
   \lambda_{ij} &= \beta_{ikjl} I^{kl} \\
   \beta_{ikjl} &= \beta \widetilde{C}_{ik} \widetilde{M}_{jl} \\
   \widetilde{C}_{ij} &= \frac{C_{ij}}{\sum_k C_{ik}} \\
   \widetilde{M}_{ij} &= \frac{M_{ij}}{\sum_k M_{ik}}
   \end{align*}
   
   TODO mention calculation of ifr from cfr, case report and asym
   
   TODO chr is being applied to asym? wouldnt that make it IHR?
   
   TODO this is missing normalization related stuff
   
   \begin{align*}
   \frac{dS_{ij}}{dt} &= - \frac{\lambda_{ij} S_{ij}}{N_{ij}} \\
   \frac{dE_{ij}}{dt} &= \frac{\lambda_{ij} S_{ij}}{N_{ij}} - \sigma E_{ij}\\
   \frac{dI^{\text{asym}}_{ij}}{dt} &= \asym(1-\chr)\sigma E_{ij} - \gamma I^{\text{asym}}_{ij}\\
   \frac{dI^{\text{mild}}_{ij}}{dt} &= (1-\asym)(1-\chr)\sigma E_{ij} - \gamma I^{\text{mild}}_{ij}\\
   \\
   \frac{dI^{\text{hosp}}_{ij}}{dt} &= \chr\sigma E_{ij} - \gamma I^{\text{hosp}}_{ij}\\
   \frac{dR^{\text{hosp}}_{ij}}{dt} &= (1-\cfr)\gamma I^{\text{hosp}}_{ij} - \rho R^{\text{hosp}}_{ij} \\
   \frac{dD_{ij}}{dt} &= \cfr\gamma I^{\text{hosp}}_{ij} \\
   \\
   \frac{dR_{ij}}{dt} &= \rho R^{\text{hosp}}_{ij} + \gamma (I^{\text{asym}}_{ij} + I^{\text{mild}}_{ij})
   \end{align*}
   


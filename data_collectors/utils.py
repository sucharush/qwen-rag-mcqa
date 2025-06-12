UNICODE_TO_LATEX = {
    # Logic and set theory
    "\u2200": r"\forall", "\u2203": r"\exists", "\u2205": r"\emptyset", "\u2207": r"\nabla",
    "\u2208": r"\in", "\u2209": r"\notin", "\u220B": r"\ni", "\u22A2": r"\vdash", "\u22A3": r"\dashv",
    "\u22A4": r"\top", "\u22A5": r"\perp", "\u2227": r"\land", "\u2228": r"\lor", "\u2201": r"\complement",
    "\u2229": r"\cap", "\u222A": r"\cup", "\u2220": r"\angle", "\u222B": r"\int",

    # Arrows
    "\u2190": r"\leftarrow", "\u2191": r"\uparrow", "\u2192": r"\rightarrow", "\u2193": r"\downarrow",
    "\u2194": r"\leftrightarrow", "\u21D0": r"\Leftarrow", "\u21D2": r"\Rightarrow",
    "\u21D4": r"\Leftrightarrow", "\u21A6": r"\mapsto",

    # Greek letters
    "\u03b1": r"\alpha", "\u03b2": r"\beta", "\u03b3": r"\gamma", "\u03b4": r"\delta", "\u03b5": r"\epsilon",
    "\u03b7": r"\eta", "\u03bb": r"\lambda", "\u03bc": r"\mu", "\u03bd": r"\nu", "\u03c0": r"\pi",
    "\u03c1": r"\rho", "\u03c3": r"\sigma", "\u03c4": r"\tau", "\u03c6": r"\phi", "\u03c7": r"\chi",
    "\u03c8": r"\psi", "\u03c9": r"\omega", "\u03d5": r"\varphi", "\u03f5": r"\varepsilon",
    "\u0393": r"\Gamma", "\u0394": r"\Delta", "\u03a0": r"\Pi", "\u03a3": r"\Sigma", "\u03a9": r"\Omega",

    # Operators
    "\u2211": r"\sum", "\u220F": r"\prod", "\u2202": r"\partial", "\u2295": r"\oplus", "\u2297": r"\otimes",
    "\u2299": r"\odot", "\u2294": r"\sqcup", "\u223c": r"\sim", "\u2212": r"-", "\u00b1": r"\pm",
    "\u2218": r"\circ", "\u2219": r"\bullet",

    # Relations
    "\u2282": r"\subset", "\u2283": r"\supset", "\u2286": r"\subseteq", "\u2287": r"\supseteq",
    "\u2284": r"\nsubset", "\u2288": r"\nsubseteq", "\u2289": r"\nsupseteq", "\u2248": r"\approx",
    "\u2260": r"\neq", "\u2264": r"\leq", "\u2265": r"\geq", "\u2261": r"\equiv", "\u226A": r"\ll",
    "\u226B": r"\gg", "\u2245": r"\cong", "\u2254": r"\coloneqq",

    # Special sets (mathbb)
    "\u2115": r"\mathbb{N}", "\u211D": r"\mathbb{R}", "\u2124": r"\mathbb{Z}", "\u2102": r"\mathbb{C}",

    # Brackets and symbols
    "\u27E8": r"\langle", "\u27E9": r"\rangle", "\u22C5": r"\cdot", "\u22C6": r"\star",
    "\u2032": r"'", "\u2033": r"''", "\u2034": r"'''", "\u00D7": r"\times"
}

EXCLUDE_SECTIONS = {
    "Etymology",
    "Background",
    "History",
    "Development",
    "Education",
    "Profession",
    "Journals",
    "Conferences",
    "Future",
    # "Applications",
    "See also",
    "References",
    "Sources",
    "Further reading",
    "External links",
    "Citations",
    "General and cited sources",  
    "Notes",
    "Bibliography",
    "Related fields",
    "Society",
    "Culture",
    "Social",
    "Cultural",
    "Economics",
    "regulation",
    "Images",
    "Visualization",
    "Extensions",
}

math_topics_by_module = {
    "Calculus (Single & Multivariable)": [
        "Limit of a function", "Continuity", "Derivative", "Chain rule",
        "Implicit function theorem", "Mean value theorem",
        "Taylor series", "L'Hôpital's rule", "Definite integral",
        "Fundamental theorem of calculus", "Improper integral",
        "Partial derivative", "Gradient", "Divergence", "Curl (mathematics)",
        "Double integral", "Triple integral", "Change of variables",
        "Jacobian matrix and determinant", "Surface integral", "Line integral",
        "Green's theorem", "Stokes' theorem", "Divergence theorem"
    ],
    "Analytic Geometry": [
        "Cartesian coordinate system", "Conic section", "Circle",
        "Ellipse", "Parabola", "Hyperbola", "Polar coordinate system",
        "Parametric equation", "Angle between vectors",  "Cylindrical coordinates", 
        "Spherical coordinate system"
    ],
    "Linear Algebra": [
        "Linear algebra", "Matrix decomposition", "Eigenvalues and eigenvectors",
        "Singular value decomposition", "Orthogonal matrix", "Least squares",
        "Change of basis", "Diagonalizable matrix", "QR decomposition",
        "Jordan normal form", "Matrix exponential", "Spectral theorem", "Pseudoinverse",
        "Kronecker product", "Block matrix", "Schur decomposition", "Linear independence",
        "Basis (linear algebra)", "Linear transformation", "Rank-nullity theorem",
        "Dual space", "Tensor product", "Minimal polynomial", "Annihilator (linear algebra)", 
        "Matrix similarity", "Generalized eigenvector", "Invariant subspace", "Cayley-Hamilton theorem"
    ],
    "Real & Functional Analysis": [
        "Calculus", "Multivariable calculus", "Partial derivative", "Gradient",
        "Taylor series", "Real analysis", "Uniform convergence", "Improper integral",
        "Mean value theorem", "Lebesgue integration", "Measure theory",
        "Banach fixed-point theorem", "Lipschitz continuity", "Sobolev space",
        "Weak convergence", "Hahn-Banach theorem", "Riesz representation theorem",
        "Compact operator", "Banach-Alaoglu theorem", "Open mapping theorem",
        "Closed graph theorem", "Compact embedding", "Lp space",
        "Duality (functional analysis)", "Continuous linear operator", "Weak topology"
    ],
    "Differential Equations": [
        "Ordinary differential equation", "Partial differential equation",
        "Method of characteristics", "Separation of variables", "Initial value problem",
        "Laplace transform", "Fourier transform"
    ],
    "Numerical Methods": [
        "Numerical analysis", "Newton's method", "Bisection method",
        "Numerical integration", "Runge-Kutta methods", "Finite difference method",
        "Condition number", "Floating-point arithmetic", "Iterative method",
        "Conjugate gradient method", "Preconditioner", "GMRES", "Arnoldi iteration",
        "Lanczos algorithm", "QR algorithm", "Jacobi method", "Multigrid method",
        "Stability (numerical analysis)", "Round-off error", "Conditioning (numerical analysis)",
        "Eigenvalue algorithm", "Inverse iteration", "Householder transformation"
    ],
    "Probability & Stochastic": [
        "Probability theory", "Random variable", "Probability distribution",
        "Expected value", "Central limit theorem", "Law of large numbers",
        "Moment generating function", "Confidence interval", "Hypothesis testing",
        "Stochastic process", "Markov chain", "Martingale", "Brownian motion",
        "Ito calculus", "Fokker-Planck equation", "Girsanov theorem"
    ],
    "Abstract Algebra": [
        "Group theory", "Ring theory", "Field (mathematics)", "Vector space",
        "Homomorphism", "Galois theory", "Polynomial ring", "Group (mathematics)",
        "Ring (mathematics)", "Module (mathematics)", "Isomorphism", "Normal subgroup",
        "Quotient group", "Direct product", "Cyclic group", "Permutation group",
        "Commutative ring", "Ideal (ring theory)", "Principal ideal domain",
        "Integral domain", "Field extension", "Galois group", "Noetherian ring",
        "Sylow theorems", "Representation theory", "Artin-Wedderburn theorem",
        "Simple group", "Automorphism group", "Abelian group", "Ring homomorphism",
        "Jacobson radical", "Z-module", "Hilbert's basis theorem", "Dedekind domain"
    ],
    "Complex Analysis": [
        "Complex analysis", "Holomorphic function", "Cauchy-Riemann equations",
        "Residue theorem", "Laurent series", "Contour integration"
    ],
    "Optimization": [
        "Convex function", "Convex optimization", "Linear programming",
        "Duality (optimization)", "Simplex algorithm", "Karush-Kuhn-Tucker conditions"
    ],
    "Topology": [
        "Topology", "Metric space", "Open set", "Compact space",
        "Continuous function (topology)", "Connected space", "Fundamental group",
        "Urysohn's lemma", "Tychonoff theorem", "Homotopy", "Covering space",
        "Manifold", "CW complex", "Simplicial complex", "Basis (topology)",
        "Separation axioms", "Product topology", "Quotient topology",
        "Path-connected space", "Homology", "Topological manifold"
    ],
    "Differential Geometry": [
        "Differentiable manifold", "Tangent bundle", "Riemannian metric",
        "Geodesic", "Connection (mathematics)", "Curvature tensor"
    ],
    "Foundations & Logic": [
        "Propositional logic", "First-order logic", "Predicate (mathematical logic)",
        "Logical connective", "Quantifier (logic)", "Boolean algebra (structure)",
        "Set theory", "Zermelo-Fraenkel set theory", "Axiom of choice",
        "Russell's paradox", "Peano axioms", "Gödel's incompleteness theorems",
        "Compactness theorem", "Löwenheim-Skolem theorem", "Constructive logic",
        "Intuitionistic logic"
    ],
    "Combinatorics & Discrete": [
        "Generating function", "Pólya enumeration theorem", "Extremal graph theory",
        "Ramsey theory", "Matroid theory"
    ],
    "Number Theory": [
        "Modular arithmetic", "Prime number", "Euclidean algorithm",
        "Chinese remainder theorem", "Fermat's little theorem", "Euler's theorem",
        "Continued fraction", "Algebraic number", "Diophantine equation",
        "Quadratic reciprocity", "Modular form", "Dirichlet character"
    ],
    "Category Theory": [
        "Category theory", "Functor", "Natural transformation", "Monoid",
        "Commutative diagram", "Yoneda lemma", "Natural isomorphism",
        "Adjoint functor", "Initial object", "Terminal object"
    ]
}
cs_topics_by_module = {
    "Foundations & Theory": [
        "Computer science", "Algorithms", "Algorithm design", "Computational complexity theory",
        "Computability theory", "Turing machine", "P vs NP problem", "Reduction (complexity)",
        "NP-completeness", "Cook-Levin theorem", "Savitch's theorem", "Time hierarchy theorem",
        "Space complexity", "Log-space reduction", "Kolmogorov complexity", "Randomized algorithm",
        "Formal language", "Automata theory", "Chomsky hierarchy"
    ],

    "Algorithms & Data Structures": [
        "Data structure", "Dynamic programming", "Amortized analysis", "Suffix tree", "Bloom filter",
        "Disjoint-set data structure", "Treap", "Binary search tree", "Red-black tree", "AVL tree",
        "Heap (data structure)", "Hash table", "Graph traversal", "Topological sorting",
        "Dijkstra's algorithm", "Bellman-Ford algorithm", "Floyd-Warshall algorithm",
        "Sorting algorithm", "Search algorithm", "Trie", "Segment tree", "Fenwick tree"
    ],

    "Formal Methods & Logic": [
        "Formal methods", "Model checking", "Hoare logic", "Temporal logic", "Automated reasoning",
        "Propositional logic", "First-order logic", "Mathematical logic", "Sequent calculus",
        "Lambda calculus", "Curry-Howard correspondence", "Big-step semantics",
        "Denotational semantics", "Type theory", "Dependent type"
    ],

    "Programming Languages & Compilers": [
        "Programming language theory", "Programming paradigm", "Formal semantics of programming languages",
        "Compiler", "Interpreter (computing)", "Type system", "Garbage collection (computer science)",
        "Syntax-directed translation", "Abstract syntax tree", "LL parser", "LR parser",
        "Static typing", "Dynamic typing", "Type inference"
    ],

    "Systems & Architecture": [
        "Operating system", "Computer architecture", "Instruction set architecture", "Instruction pipeline",
        "Cache (computing)", "Cache coherence", "Thread (computing)", "Context switch", "Segmentation fault",
        "Concurrency (computer science)", "Multithreading", "Parallel computing", "Virtual memory",
        "Page replacement algorithm", "Memory consistency model", "Scheduling (computing)", "Deadlock",
        "Race condition", "Mutex", "Semaphore", "Spinlock", "System call"
    ],

    "Distributed & Parallel Computing": [
        "Distributed computing", "Shared memory", "Message passing", "Concurrency control",
        "MapReduce", "Bulk synchronous parallel", "Paxos algorithm", "Raft consensus algorithm",
        "CAP theorem", "Vector clock", "Gossip protocol", "Eventual consistency", "Leader election"
    ],

    "Software Engineering & Tooling": [
        "Software engineering", "Software testing", "Software design pattern",
        "Integrated development environment", "Version control", "Debugging", "Code coverage",
        "Static analysis", "Test-driven development", "Continuous integration"
    ],

    "Databases & Storage Systems": [
        "Database", "Relational database", "SQL", "Normalization (database)",
        "Index (database)", "Transaction processing", "ACID", "NoSQL", "CAP theorem", "Sharding",
        "Query optimization"
    ],

    "Security & Cryptography": [
        "Cryptography", "Public-key cryptography", "Symmetric-key algorithm", "Asymmetric-key algorithm",
        "Cryptographic hash function", "Digital signature", "RSA algorithm", "Elliptic curve cryptography",
        "Zero-knowledge proof", "Homomorphic encryption", "Diffie-Hellman key exchange",
        "Lattice-based cryptography", "Authentication", "Authorization", "TLS", "Man-in-the-middle attack"
    ],

    "Machine Learning Foundations": [
        "Artificial intelligence", "Machine learning", "Deep learning", "Reinforcement learning",
        "Supervised learning", "Unsupervised learning", "Gradient descent",
        "VC dimension", "PAC learning", "Bias-variance tradeoff", "Kernel method",
        "Maximum likelihood estimation", "Bayesian inference", "Stochastic gradient descent",
        "Overfitting", "Regularization (machine learning)", "Cross-validation",
        "Loss function", "Decision boundary", "Linear classifier", "Support vector machine",
        "Bayesian network"
    ],

    "Natural Language Processing & Vision": [
        "Natural language processing", "Text classification", "Information retrieval",
        "Computer vision", "Image recognition", "Bag-of-words model", "TF-IDF", "Word embedding",
        "Convolutional neural network", "Recurrent neural network", "Transformer (machine learning)"
    ],

    "Robotics & Interaction": [
        "Robotics", "Human-computer interaction", "Computer graphics",
        "Graphical user interface", "Virtual reality", "Gesture recognition"
    ],

    "Data & Computation": [
        "Numerical analysis", "Symbolic computation", "Data mining", "Big data",
        "Data compression", "Information theory", "Hash function", "Computational geometry",
        "Approximation algorithm", "Streaming algorithm", "Sketching algorithm"
    ],

    "Emerging Topics": [
        "Quantum computing", "Cloud computing", "Blockchain", "Federated learning", "Edge computing",
        "Differential privacy", "Explainable artificial intelligence"
    ]
}
ds_topics_by_module = {
    "Deep Learning Fundamentals": [
        "Backpropagation", "Batch normalization", "Dropout (neural networks)",
        "Weight initialization", "Vanishing gradient problem", "Exploding gradient problem",
        "ReLU", "Sigmoid function", "Softmax function"
    ],
    "Architectures & Models": [
        "Convolutional neural network", "Recurrent neural network", "Long short-term memory",
        "Gated recurrent unit", "Transformer (machine learning)", "Attention (machine learning)",
        "Residual neural network", "Autoencoder", "Variational autoencoder",
        "Generative adversarial network", "BERT (language model)"
    ],
    "Training & Optimization": [
        "Learning rate", "Cross-entropy", "Gradient descent", "Early stopping",
        "Data augmentation", "Transfer learning", "Gradient clipping", "Adam optimization algorithm"
    ],
    "Regularization & Generalization": [
        "Regularization (mathematics)", "Weight decay", "Generalization error",
        "Overfitting", "Bias-variance tradeoff", "Cross-validation"
    ],
    "Representation Learning": [
        "Representation learning", "Word embedding", "Word2vec", "GloVe (machine learning)",
        "Latent variable", "Principal component analysis", "Independent component analysis",
        "Canonical correlation", "Multidimensional scaling", "Factor analysis",
        "t-distributed stochastic neighbor embedding", "Uniform Manifold Approximation and Projection",
        "Dimensionality reduction"
    ],
    "Interpretability & Robustness": [
        "Adversarial machine learning", "Saliency map"
    ],
    "Theory & Foundations": [
        "Universal approximation theorem", "Empirical risk minimization", "Inductive bias",
        "VC dimension", "PAC learning", "Probably approximately correct learning",
        "Information theory", "Entropy (information theory)", "Kullback-Leibler divergence",
        "Mutual information", "Minimum description length"
    ],
    "Statistical Foundations": [
        "Statistical inference", "Confidence interval", "Hypothesis testing",
        "p-value", "Statistical power", "Multiple comparisons problem",
        "Likelihood function", "Maximum likelihood estimation", "Bayesian inference",
        "Credible interval", "Prior probability", "Posterior probability",
        "Bayes' theorem"
    ],
    "Distributions": [
        "Probability distribution", "Normal distribution", "Binomial distribution",
        "Poisson distribution", "Exponential distribution", "Multivariate normal distribution"
    ],
    "Regression & Models": [
        "Linear regression", "Logistic regression", "Ridge regression", "Lasso (statistics)",
        "Generalized linear model"
    ],
    "Evaluation & Metrics": [
        "Confusion matrix", "Receiver operating characteristic", "Precision and recall",
        "F1 score", "Area under the curve", "Sensitivity and specificity"
    ],
    "Bayesian & Sampling": [
        "Markov chain Monte Carlo", "Gibbs sampling", "Rejection sampling",
        "Metropolis-Hastings algorithm"
    ],
    "Clustering & Unsupervised Learning": [
        "K-means clustering", "Hierarchical clustering", "Gaussian mixture model",
        "Silhouette (clustering)", "DBSCAN"
    ],
    "Feature Engineering": [
        "Feature selection", "One-hot encoding", "Missing data", "Imputation (statistics)",
        "Outlier", "Standard score"
    ],
    "Time Series & Forecasting": [
        "Time series", "Autoregressive model", "Moving average", "ARIMA",
        "Stationary process", "Decomposition of time series", "Forecasting", "Exponential smoothing"
    ],
    "Causal Inference & Experimental Design": [
        "Design of experiments", "Randomized controlled trial", "Blocking (statistics)",
        "Confounding", "Instrumental variable", "Propensity score matching",
        "Difference in differences", "Regression discontinuity design"
    ]
}
phy_topics_by_module = {
    "Classical Mechanics": [
        "Displacement", "Velocity", "Acceleration", "Uniform circular motion",
        "Newton's first law", "Newton's second law", "Newton's third law",
        "Friction", "Kinetic friction", "Static friction", "Tension (physics)",
        "Normal force", "Work (physics)", "Power (physics)", "Kinetic energy",
        "Potential energy", "Mechanical energy", "Conservation of energy",
        "Linear momentum", "Conservation of momentum", "Impulse (physics)",
        "Elastic collision", "Inelastic collision",
        "Torque", "Angular momentum", "Moment of inertia", "Rotational kinetic energy",
        "Rolling motion", "Center of mass", "Equilibrium (physics)",
        "Simple harmonic motion", "Hooke's law", "Damped oscillation",
        "Resonance", "Pendulum", "Projectile motion", "Inclined plane"
    ],

    "Waves & Optics": [
        "Transverse wave", "Longitudinal wave", "Wavelength", "Frequency",
        "Wave speed", "Superposition principle", "Standing wave", "Resonant frequency",
        "Sound wave", "Doppler effect", "Beats (acoustics)",
        "Reflection (physics)", "Refraction", "Snell's law", "Total internal reflection",
        "Dispersion (optics)", "Diffraction", "Interference (wave propagation)",
        "Young's double-slit experiment", "Thin film interference",
        "Polarization (waves)", "Geometrical optics", "Convex lens", "Concave lens",
        "Mirror equation", "Lens formula", "Magnification"
    ],

    "Thermodynamics & Statistical Mechanics": [
        "Zeroth law of thermodynamics", "First law of thermodynamics",
        "Second law of thermodynamics", "Third law of thermodynamics",
        "Internal energy", "Heat", "Work (thermodynamics)",
        "Specific heat capacity", "Latent heat", "Heat transfer",
        "Conduction", "Convection", "Radiation",
        "Carnot engine", "Efficiency (thermodynamics)", "Entropy", "Enthalpy",
        "Ideal gas law", "Boltzmann constant", "Thermal equilibrium",
        "Thermal expansion", "Maxwell-Boltzmann distribution", "Partition function"
    ],

    "Electricity & Magnetism": [
        "Coulomb's law", "Electric field", "Electric potential", "Voltage",
        "Capacitance", "Parallel plate capacitor", "Dielectric", "Ohm's law",
        "Resistors in series", "Resistors in parallel", "Kirchhoff's current law",
        "Kirchhoff's voltage law", "Power (electricity)", "Electromotive force",
        "RC circuit", "Current", "Drift velocity",
        "Magnetic field", "Lorentz force", "Biot-Savart law", "Ampère's law",
        "Faraday's law of induction", "Lenz's law", "Inductance", "Transformer",
        "AC circuit", "RLC circuit", "Maxwell's equations"
    ],

    "Modern Physics": [
        "Speed of light", "Photoelectric effect", "Planck's constant", "Photon energy",
        "Compton scattering", "Wave-particle duality", "De Broglie wavelength",
        "Heisenberg uncertainty principle", "Schrödinger equation",
        "Quantum tunneling", "Energy level", "Electron orbital", "Spin (physics)",
        "Pauli exclusion principle", "Bohr model", "Zeeman effect",
        "Relativity", "Time dilation", "Length contraction", "Mass-energy equivalence",
        "E = mc^2", "Twin paradox", "Muon decay", "Rest mass"
    ]
}
eng_topics_by_module = {
    "Electrical Circuits & Components": [
        "Resistor", "Capacitor", "Inductor", "Diode", "Transistor", "Operational amplifier",
        "Ohm's law", "Kirchhoff's circuit laws", "Thevenin's theorem", "Norton's theorem",
        "Superposition theorem", "RC circuit", "RL circuit", "RLC circuit", "Impedance",
        "Reactance", "Resonance (electric circuits)", "Voltage divider", "Current divider"
    ],
    "Control Systems Engineering": [
        "Control system", "Feedback", "Open-loop control", "Closed-loop control",
        "Proportional-integral-derivative controller", "Transfer function", "Block diagram",
        "Stability (control theory)", "Bode plot", "Nyquist stability criterion",
        "Root locus", "State-space representation", "Controllability", "Observability",
        "Pole-zero plot", "Time constant", "Settling time", "Rise time", "Overshoot"
    ],
    "Signal Processing": [
        "Signal processing", "Analog signal", "Digital signal", "Sampling (signal processing)",
        "Quantization (signal processing)", "Nyquist-Shannon sampling theorem",
        "Aliasing", "Fourier transform", "Discrete Fourier transform", "Fast Fourier transform",
        "Convolution", "Correlation", "Filter (signal processing)", "Low-pass filter",
        "High-pass filter", "Band-pass filter", "Band-stop filter", "Impulse response",
        "Frequency response", "Signal-to-noise ratio"
    ],
    "Digital Image Processing": [
        "Digital image", "Pixel", "Grayscale", "Color space", "Histogram (image processing)",
        "Histogram equalization", "Thresholding (image processing)", "Edge detection",
        "Sobel operator", "Canny edge detector", "Gaussian blur", "Median filter",
        "Morphological operations", "Dilation (morphology)", "Erosion (morphology)",
        "Opening (morphology)", "Closing (morphology)", "Image segmentation",
        "Region growing", "Watershed algorithm"
    ],
    "Instrumentation & Measurement": [
        "Sensor", "Transducer", "Strain gauge", "Thermocouple", "Photodiode",
        "Piezoelectric sensor", "Signal conditioning", "Analog-to-digital converter",
        "Digital-to-analog converter", "Resolution (measurement)", "Accuracy and precision",
        "Calibration", "Error analysis", "Noise (electronics)", "Signal-to-noise ratio",
        "Bandwidth (signal processing)", "Sampling rate", "Quantization error"
    ],
    "Power Systems & Electrical Machines": [
        "Power engineering", "Electric power transmission", "Electric power distribution",
        "Transformer", "Three-phase electric power", "Synchronous motor", "Induction motor",
        "Direct current motor", "Alternator", "Load flow analysis", "Short-circuit analysis",
        "Per-unit system", "Circuit breaker", "Protective relay", "Ground fault",
        "Power factor", "Reactive power", "Harmonics (electrical power)"
    ],
    "Electronics & Semiconductor Devices": [
        "Semiconductor", "PN junction", "Zener diode", "Bipolar junction transistor",
        "Field-effect transistor", "MOSFET", "CMOS", "Amplifier", "Oscillator",
        "Voltage regulator", "Switching regulator", "Phase-locked loop", "Schmitt trigger",
        "Analog multiplier", "Digital logic gate", "Flip-flop (electronics)",
        "Multiplexer", "Demultiplexer", "Analog-to-digital converter", "Digital-to-analog converter"
    ],
    "Communication Systems": [
        "Modulation", "Amplitude modulation", "Frequency modulation", "Phase modulation",
        "Pulse-code modulation", "Quadrature amplitude modulation", "Digital modulation",
        "Demodulation", "Bandwidth (signal processing)", "Signal-to-noise ratio",
        "Bit error rate", "Channel capacity", "Shannon-Hartley theorem", "Multiplexing",
        "Time-division multiplexing", "Frequency-division multiplexing", "Orthogonal frequency-division multiplexing",
        "Error detection and correction", "Hamming code", "Cyclic redundancy check"
    ]
}
chem_topics_by_module = {
    "Atomic & Molecular Structure": [
        "Atom", "Electron", "Proton", "Neutron", "Isotope",
        "Atomic number", "Mass number", "Mole (unit)", "Avogadro constant",
        "Electron configuration", "Quantum number", "Orbital hybridisation",
        "Atomic orbital", "Molecular orbital", "Bond order", "Ionization energy",
        "Electronegativity", "Electron affinity"
    ],
    "Chemical Bonding & Interactions": [
        "Ionic bond", "Covalent bond", "Polar covalent bond", "Coordinate bond",
        "Metallic bond", "Intermolecular force", "Hydrogen bond",
        "Van der Waals force", "Dipole-dipole interaction", "London dispersion force"
    ],
    "Stoichiometry & Reactions": [
        "Chemical formula", "Empirical formula", "Molecular formula",
        "Balancing chemical equations", "Stoichiometry", "Limiting reagent",
        "Percent yield", "Theoretical yield", "Combustion", "Synthesis reaction",
        "Decomposition reaction", "Single displacement reaction", "Double displacement reaction"
    ],
    "Equilibrium & Thermodynamics": [
        "Chemical equilibrium", "Equilibrium constant", "Le Chatelier's principle",
        "Reaction quotient", "Gibbs free energy", "Enthalpy", "Entropy",
        "Thermochemistry", "Laws of thermodynamics", "Thermodynamic equilibrium"
    ],
    "Acids, Bases, and Electrochemistry": [
        "Acid-base reaction", "Arrhenius acid", "Brønsted-Lowry acid", "Lewis acid",
        "pH", "pKa", "Buffer solution", "Neutralization", "Titration",
        "Electrochemistry", "Redox", "Oxidation state", "Electrode potential",
        "Galvanic cell", "Electrolysis", "Nernst equation", "Fuel cell", "Battery (electricity)", "Corrosion"
    ],
    "Kinetics & Catalysis": [
        "Reaction rate", "Rate law", "Order of reaction", "Activation energy",
        "Catalyst", "Inhibitor", "Reaction mechanism", "Intermediate (chemistry)",
        "Transition state", "Arrhenius equation"
    ],
    "Organic Chemistry": [
        "Hydrocarbon", "Alkane", "Alkene", "Alkyne", "Aromatic compound",
        "Alcohol", "Aldehyde", "Ketone", "Carboxylic acid", "Amine",
        "Ester", "Amide", "Nitrile", "Halide",
        "Isomer", "Structural isomer", "Stereoisomer", "Chirality",
        "SN1 reaction", "SN2 reaction", "E1 reaction", "E2 reaction",
        "Electrophilic addition", "Nucleophilic substitution", "Free radical mechanism"
    ],
    "Inorganic Chemistry": [
        "Transition metal", "Coordination complex", "Ligand", "Crystal field theory",
        "Chelate", "Oxidation number", "Acid-base chemistry", "Metal oxide",
        "Hydrolysis", "Salt (chemistry)"
    ],
    "Analytical Techniques": [
        "Gravimetric analysis", "Volumetric analysis", "Titration",
        "Spectrophotometry", "Chromatography", "Gas chromatography",
        "Mass spectrometry", "Nuclear magnetic resonance spectroscopy",
        "Infrared spectroscopy", "Ultraviolet-visible spectroscopy"
    ]
}
bio_med_topics_by_module = {
    "Cell & Molecular Biology": [
        "Cell (biology)", "Cell membrane", "Organelle", "Nucleus (cell)", "Mitochondrion",
        "Ribosome", "Endoplasmic reticulum", "Golgi apparatus", "Cytoskeleton", "Lysosome",
        "Cytoplasm", "Plasma membrane", "DNA", "RNA", "Gene", "Gene expression",
        "Transcription (biology)", "Translation (biology)", "DNA replication",
        "Mutation", "Genetic code", "Chromosome", "Chromatin", "Nucleosome",
        "Cell cycle", "Mitosis", "Meiosis", "Apoptosis"
    ],

    "Genetics & Inheritance": [
        "Mendelian inheritance", "Genotype", "Phenotype", "Allele", "Dominance (genetics)",
        "Recessive allele", "Genetic recombination", "Crossing over", "Linkage (genetics)",
        "Pedigree chart", "Punnett square", "Heredity", "Sex-linked trait"
    ],

    "Biochemistry & Metabolism": [
        "Amino acid", "Protein", "Peptide bond", "Enzyme", "Substrate (biology)",
        "Enzyme kinetics", "ATP", "Metabolism", "Glycolysis", "Citric acid cycle",
        "Electron transport chain", "Oxidative phosphorylation", "Fermentation",
        "Photosynthesis", "Chloroplast", "Lipids", "Carbohydrate", "Nucleic acid"
    ],

    "Human Anatomy & Physiology": [
        "Nervous system", "Neuron", "Synapse", "Neurotransmitter", "Endocrine system",
        "Hormone", "Hypothalamus", "Pituitary gland", "Circulatory system", "Heart",
        "Blood", "Blood pressure", "Red blood cell", "White blood cell",
        "Respiratory system", "Lung", "Gas exchange", "Digestive system", "Liver",
        "Pancreas", "Kidney", "Nephron", "Urinary system", "Immune system", "Skin",
        "Skeletal system", "Muscle", "Muscular system", "Homeostasis", "Thermoregulation"
    ],

    "Microbiology & Immunology": [
        "Bacteria", "Virus", "Fungus", "Pathogen", "Antigen", "Antibody",
        "Innate immune system", "Adaptive immune system", "Inflammation",
        "Complement system", "Cytokine", "Vaccination", "Immunological memory",
        "Antibiotic", "Antiviral drug", "Antifungal", "Vaccine"
    ],

    "Molecular Techniques": [
        "Polymerase chain reaction", "DNA sequencing", "Gel electrophoresis",
        "Gene editing", "CRISPR", "Cloning", "Western blot", "Northern blot",
        "Southern blot", "ELISA", "Microarray", "Restriction enzyme", "cDNA"
    ],

    "Medical Conditions & Diagnostics": [
        "Pathology", "Hypertension", "Diabetes mellitus", "Cancer", "Asthma",
        "Stroke", "Heart failure", "Infectious disease", "Electrocardiogram",
        "Biopsy", "Radiology", "MRI", "CT scan", "Blood test", "Anemia"
    ],

    "Development & Evolution": [
        "Embryology", "Zygote", "Blastocyst", "Gastrulation", "Stem cell",
        "Differentiation (biology)", "Morphogenesis", "Evolution", "Natural selection",
        "Speciation", "Phylogenetics", "Molecular evolution"
    ],

    "Basic Ecology": [
        "Ecology", "Ecosystem", "Population ecology", "Community ecology",
        "Food chain", "Food web", "Carbon cycle", "Nitrogen cycle",
        "Energy pyramid", "Biotic factor", "Abiotic factor"
    ]
}
urls = [
    "https://stanford.edu/~shervine/teaching/cme-106/cheatsheet-probability",
    "https://stanford.edu/~shervine/teaching/cme-106/cheatsheet-statistics",
    "https://stanford.edu/~shervine/teaching/cme-106/key-concepts",
    "https://stanford.edu/~shervine/teaching/cme-102/linear-algebra",
    "https://stanford.edu/~shervine/teaching/cme-102/calculus",
    "https://stanford.edu/~shervine/teaching/cme-102/trigonometry",
    "https://stanford.edu/~shervine/teaching/cme-102/cheatsheet-first-ode",
    "https://stanford.edu/~shervine/teaching/cme-102/cheatsheet-second-ode",
    "https://stanford.edu/~shervine/teaching/cme-102/cheatsheet-applications",
    "https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-reflex-models",
    "https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-states-models",
    "https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-variables-models",
    "https://stanford.edu/~shervine/teaching/cs-221/cheatsheet-logic-models",
    "https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning",
    "https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-unsupervised-learning",
    "https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning",
    "https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks",
    "https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks",
    "https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks",
    "https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks"
]


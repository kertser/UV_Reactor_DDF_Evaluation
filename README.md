# DDF Evaluation and Reactor Efficiency Analysis

This project aims to evaluate the dose distribution function (DDF) of a hydrophotonic reactor and determine its efficiency using various statistical measures and optimizations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project uses the inverse gamma distribution to model the dose distribution in a hydrophotonic reactor. It evaluates the reactor's efficiency based on dimensionless factors such as the Dose Spread Level (DSL), Coefficient of Variation (CV), and Tail Coefficient of Variation (TCV).

## Features

- Load and preprocess DDF data
- Calculate empirical moments (mean, variance, skewness, kurtosis)
- Optimize inverse gamma distribution parameters
- Calculate dimensionless efficiency factors
- Plot dose distribution and fitted distribution
- Calculate and display overall reactor efficiency

## Setup

### Prerequisites

- Python 3.7 or higher
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/DDF_eval.git
    cd DDF_eval
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Place your `DDF.csv` file in the project directory.

## Usage

1. Run the main script:

    ```bash
    python src/main.py
    ```

2. The script will perform the following:
    - Load and preprocess the DDF data.
    - Calculate empirical moments of the data.
    - Perform a grid search to find the best initial parameters for the inverse gamma distribution.
    - Optimize the parameters of the inverse gamma distribution.
    - Calculate dimensionless factors (DSL, CV, TCV) and overall efficiency.
    - Plot the dose distribution and the fitted inverse gamma distribution.
    - Display the calculated parameters and efficiency metrics.

## Results

The script will output:
- Optimized parameters of the inverse gamma distribution (alpha, beta, location).
- Empirical and fitted moments (mean, variance, skewness, kurtosis).
- Dimensionless factors: DSL, CV, TCV.
- Overall efficiency of the reactor as a percentage.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

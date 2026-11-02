Implementation of the paper "Federated Learning for Critical Electrical Infrastructure - Handling Non-IID Data for Predictive Maintenance of Substation Equipment" by Soham Ghosh and Gaurav Mittal.

All work must be cited as follows: *Ghosh S and Mittal G (2026) Federated learning for critical electrical infrastructure - handling data heterogeneity for predective maintenance of substation equipment. Front. Artif. Intell 8:1697175*

There are three datasets that are provided in this repository:

1. HV Circuit Breaker Maintenance Data (Excel file)

    -   The dataset is derived by combining results from high-voltage breaker monitoring at utility substations, where sensors capture critical measurements such as SF6 density, breaker status, and ambient cabinet temperature. Note: This dataset is semi-synthetically derived for research and educational purposes and may not fully represent real-world operational data.
    -   Utility engineers at participating utilities routinely review breaker records and associated measurements for assets under their operational oversight and flag cases requiring maintenance. These inspection-driven flags are typically based on factors such as SF6 dew point, fault operation count, contact life, and days since last operation, in accordance with each utility’s established maintenance standards.
    -   The dataset comprises 5,000 samples of breaker readings, aggregated from five representative utilities. Each breaker is assigned a unique categorical identifier corresponding to its source utility, with 28 features representing various sensor-based measurements and one quality metric (0: no maintenance required, 1: maintenance required).
    -   It is important to note that not all 28 features are uniformly available across all product variants, leading to slight variations in the feature space between variants.
    -   To enhance the training of machine learning models, the dataset has been augmented with an increased proportion of negative examples to ensure sufficient representation of failure cases.

2. Large Power Transformer Maintenance Data (Excel file)

    -   The dataset is similar to the HV Circuit Breaker Maintenance Data Set, and is derived by combining results from large power transformer monitoring at utility substations, where sensors capture critical measurements such as LTC and main tank oil temperature, dissolved gas values (in ppm). Note: This dataset is semi-synthetically derived for research and educational purposes and may not fully represent real-world operational data.
    -   Utility engineers at participating utilities routinely review transformer records and associated measurements for assets under their operational oversight and flag cases requiring maintenance. These inspection-driven flags are typically based on factors such as high oil temperature, and excessive amount of certain dissolved gas (usually based on Duval triangles and pentagons), following each utility’s established maintenance standards.


3. Emergency Station Generator Maintenance Data (Excel file)

     -   The dataset is derived by combining results from station emergency generators at utility substations (~150 - 200 kVA on propane or natural gas), where sensors capture critical measurements such as engine temperatures, oil pressure, alternate current and voltages. Note: This dataset is semi-synthetically derived for research and educational purposes and may not fully represent real-world operational data.
     -   Utility engineers at participating utilities routinely review station emergency generator records and associated measurements for assets under their operational oversight and flag cases requiring maintenance. These inspection-driven flags are typically based on factors such as start attempts, battery state of charge, lube oil temperature, crankcase pressure, and emission levels, in accordance with each utility’s established maintenance standards.

Additional notes: 
$\textcolor{red}{DISCLAIMER:}$
These datasets are semi-synthetically derived for research and educational purposes and may not fully represent real-world operational data. The databases are provided "as is" without any express or implied warranties, including, but not limited to, implied warranties of merchantability and fitness for a particular purpose. In no event shall we (the authors) be liable for any direct, indirect, incidental, special, exemplary, or consequential damages, however caused and under any theory of liability, whether in contract, strict liability, tort (including negligence or otherwise), or otherwise, arising in any way out of the use of the database. By downloading, accessing, or using the database, you hereby release us from any and all liability that may arise from your use of the database. 

Given that these databases are intended for educational and research purposes only, it is imperative that these are not used as a substitute for real-world electrical asset protection, operational decision-making, or maintenance planning.

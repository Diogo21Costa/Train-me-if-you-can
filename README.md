# Train me if you can: decentralizedlearning on the deep edge

This repository contains the code and experiments for the paper:

> Fair Resource Allocation in Federated Learning

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#SGD">SGD</a></li>
    <li><a href="#L-SGD">L-SGD</a></li>
    <li><a href="#STM32F767-ZI">STM32F767-ZI</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


## SGD
### MNIST dataset

|                           | SGD (Python) |   SGD (C)   |
|---------------------------|----------|---------|
|        Accuracy (%)       |   94.76  |  90.65  |
| Memory footprint (KBytes) |  135.632 | 135.632 |
|     Latency (s/epoch)     |   8.25   |  8.362  |

### Drowsiness detection dataset


|                           | SGD (Python) |   SGD (C)   |
|---------------------------|----------|---------|
|        Accuracy (%)       |   82.20  |  83.51  |
| Memory footprint (KBytes) |   6.816  |  6.816  |
|     Latency (s/epoch)     |   2.300  | 0.1512  |

## L-SGD

### MNIST dataset

|                           | SGD (Python) |   SGD (C)   |   L-SGD (C)   |
|---------------------------|----------|---------|---------|
|        Accuracy (%)       |   94.76  |  90.65  |  93.54  |
| Memory footprint (KBytes) |  135.632 | 135.632 |  129.232  |
|     Latency (s/epoch)     |   8.25   |  8.362  | 2.250  |

### Drowsiness detection dataset


|                           | SGD (Python) |   SGD (C)   |   L-SGD (C)   |
|---------------------------|----------|---------|---------|
|        Accuracy (%)       |   82.20  |  83.51  |  82.18  |
| Memory footprint (KBytes) |   6.816  |  6.816  |  1.568  |
|     Latency (s/epoch)     |   2.300  | 0.1512  | 0.0943  |

## STM32F767-ZI

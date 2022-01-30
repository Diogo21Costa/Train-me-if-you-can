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

                            |        MNIST       |  Private Dataset  |
                            |----------------------------------------|
|                           |    SGD   |   SGD   |    SGD   |   SGD  |
|                           | (Python) |   (C)   | (Python) |   (C)  |
|---------------------------|----------|---------|----------|--------|
|        Accuracy (%)       |   94.76  |  90.65  |   82.20  |  83.51 |
| Memory footprint (KBytes) |  135.632 | 135.632 |   6.816  |  6.816 |
|     Latency (s/epoch)     |   8.25   |  8.362  |   2.300  | 0.1512 |

## L-SGD

## STM32F767-ZI

# Santander Product Recommendation
[Kaggle - Can you pair products with people?] (https://www.kaggle.com/c/santander-product-recommendation)

## Introduction
The goal of this competition was to predict which products Santander's existing customers will
use in the next month based on their past behavior and that of similar customers, using 1.5 years of historical data.
<br> 24 different products were included in the dataset 
and competitors were tasked with developing a recommendation system using [mean average precision @ 7] (https://www.kaggle.com/c/santander-product-recommendation/details/evaluation) as the scoring metric.

## Team Members
The team, Arrested Development, consisted of [Tyrone Cragg] (https://github.com/tyronecragg) and [Liam Culligan] (https://github.com/liamculligan).

## Performance
The solution obtained a rank of [147th out of 1785 teams] (https://www.kaggle.com/c/santander-product-recommendation/leaderboard/private)
with a private leaderboard score of 0.0303227.<br>

## Execution
1. Close this repository <br>
2. [Download the data from Kaggle] (https://www.kaggle.com/c/santander-product-recommendation/data) and place in the working directory
3. Run `pre-process.R`
5. Run the bagged XGB model script, `XGB.R`<br>

## Requirements
* R 3+

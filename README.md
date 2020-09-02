# Chasing the Whale
 

![whale_tail](https://github.com/coolwonny/Chasing-the-Whale/blob/master/Images/whale-tail.jpg)    
---
by Marcus S Kim  
---
## Project Goal
Use **SEC 13F filings** and *whalescores* to build a model based on fund performance and investor sector weightings that will outperform the S&P 500 in annual return.

## Data Sources and Analysis
- The kind of data or subject area we will use
  -	Form 13F is a quarterly report filed, per United States Securities and Exchange Commission regulations, by "institutional investment managers" with control over $100M in assets to the SEC, listing all equity assets under management
  -	Index ETFs for the S&P 500 sectors

- The kinds of questions you will ask of the data
  -	Identify top performing funds
  -	Identify investment sector weightings for top performing funds
  -	What are the most common terms/names mentioned within each investment sector
  
- One or more data sources that will meet your requirements
  -	[Whalewisdom.com](https://whalewisdom.com/)
  -	[Validea.com](https://www.validea.com/)
  -	[Sec.gov_edgar](https://www.sec.gov/edgar.shtml)
  - [Sector SPDR](https://www.sectorspdr.com/)

## About Whalewisdom
**[Whalewisdom](https://whalewisdom.com/)** is a web site providing information on hedge funds responsible for filing to SEC. It attempts to identify market managers that outperform the market on a consistent basis.  

The group assigns, [whalescores](https://whalewisdom.com/info/whalescores), to managers.  Whalescores take into consideration risk measurements and returns to identify managers most likely to outperform the market.

There are minimum criteria required to be included:
- No banks, trusts, insurance companies   
- Must manage a portfolio greater than $100 million    
- Holdings must be between 5 and 750

## About Sector ETFs   
SPDR provides unique Exchange Traded Funds (ETFs) that divide the S&P into eleven index funds traded throughout the day on NYSE Arca.

- Breakdown of industry sectors by the 11 sector ETFs:
  - Materials - **XLB**
  - Industrials, Transports - **XLI**
  - Financials - **XLF**
  - Energy - **XLE**
  - Consumer discretionary - **XLY**
  - Information technology - **XLK**
  - Communication services - **XLC**
  - Real estate - **XLRE**
  - Health care - **XLV**
  - Consumer staples - **XLP**
  - Utilities and Telecommunications - **XLU**



## Model Structure   
The basic idea is to separate modeling into two parts and put them back together when they’re done.   
> Part 1: Getting a specific weight on each sector by feeding historical weights of the most successful hedge funds (Dataset can be achieved from EDGAR and Whalewisdom)        
   

> Part 2: Getting a yearly or quarterly return for each index ETF that will represent each sector in Part 1.
At the end of the day, we can combine the results to get an idea of what weights will make the optimized sector allocation for our optimized portfolio.

### Structure  
       
> Part 1:
1.	Determine which funds to be selected for our project
(Could be top-n (3, 5, 10..) best performed ones or other standards that we may consider proper)
2.	Pulling the quarterly 13-F data from Whalewisdom for the selected funds
3.	Define the features and outputs. 
(Ideally features should be weightings on each sector quarterly that spitting out optimized weightings for output that would have given the best portfolio return over that time periods. However, building this kind of model could be difficult to implement.)

> Part 2: 
1.	Define which index ETF matching with which sector in Part 1.
2.	Determine the period that we are going to analyze (probably starting with OCT 1, 2018 when the reclassification started)
3.	Pulling each ETF’s historical data from APIs (Alpaca)
4.	Data preprocessing with the data. Need to apply the time-series way of splitting it.
(Features = daily closing price, Output = daily closing price (using rolling window))
5.	Build and Train the model using RNN LSTM model
6.	Predict the data using test dataset
7.	Evaluate the model

After completing both parts, we can simulate a portfolio by using the weights recommended from Part 1, applying them to relevant ETFs to see the result (prediction). 
Since we don’t know the exact dates when the hedge funds rebalanced their weights on each sector during a certain quarter, we need to assume that they did in the middle of the quarter(45 days after the quarter starts) for our simulated portfolio.
When we get a predicted result of return from our simulation portfolio, then we should compare the result to that of the real funds to see how relevant the result is. If it is relevant enough, we might use this method to create our real-world portfolio.

## Part 1
Part 1 is about finding the optimized weightings. Consequentially, we ended up using only two methods.
- Average weightings per investor sectors
- Best quarterly weightings quarter after quarter. Best quarterly means replicating the sector weightings of the best performing fund of that quarter. We used whalescores per quarter to pick the champions in each quarter.

Unfortunately, we could not come up with creating a set of optimized weightings through applying machine learning technique regardless of our attempts to code it with several different ML models including the RNN, regression and random forest. The goal is to have 11 targets from training 11 features of sector weightings where the sum of targets must equal to 1 or 100%. The Multivariate regression model could possibly be the one but failed to complete a code due to lack of advanced knowledge and limitation of time. 

Another approach was using `cvxpy` library. We might be using it finding the optimal weights by defining a function applying `cvx.Minimize()` as below.    

`import cvxpy as cvx`

`def get_optimal_weights(covariance_returns, index_weights, scale=2.0):`   

  ` """ `
   ` Find the optimal weights.`

   ` Parameters`   
   ` ----------`   
    `covariance_returns : 2 dimensional Ndarray`   
       ` The covariance of the returns`   
    `index_weights : Pandas Series`   
        `Index weights for all tickers at a period in time`   
    `scale : int`   
        `The penalty factor for weights the deviate from the index `   
    `Returns`   
    `-------`   
    `x : 1 dimensional Ndarray`   
    `    The solution for x`   
    `"""`   
    `assert len(covariance_returns.shape) == 2`   
    `assert len(index_weights.shape) == 1`   
    `assert covariance_returns.shape[0] == covariance_returns.shape[1]  == index_weights.shape[0]`   
    
   ` m = covariance_returns.shape[0]`   
   ` x = cvx.Variable(m)`
    
   ` objective = cvx.Minimize(cvx.quad_form(x, covariance_returns) + scale * cvx.norm(x - index_weights))`   
   ` constraints = [x >= 0, sum(x) == 1]`   
    `problem = cvx.Problem(objective, constraints)`   
   ` problem.solve()`   
       
  `return x.value`

As a result, we could not figure out how to apply the above function to generate 11 different optimized weights that becomes 100% when summing up.

## Part 2
Part 2 is mainly taking the sector weighting data from Part 1, generating daily portfolio value by putting them together with daily index ETF’s closing price. In this way, we have weighted daily return with rebalancing it every quarter by quarter. Then, we are good to calculate the weighted cumulative portfolio returns to see whether the portfolio can outperform the market return, proxied by S&P 500.
     
![etf returns](https://github.com/coolwonny/Chasing-the-Whale/blob/master/Images/ETFs_returns.png)   

From the picture above, you may see the cumulative returns of 11 different index ETF’s over the last two years. You can see the brown line which is **XLK** representing **IT** and it especially performs well since the 3Q2019 while the green line **XLE** is from **Energy** sector that performed poorly in the same time frame. We’re creating a portfolio using only these 11 ETFs with different weightings to find out whether we can catch up the outperforming hedge funds.

![berkshire](https://github.com/coolwonny/Chasing-the-Whale/blob/master/Images/berkshire_portfolios.png)

The above plot shows the approach made in Part 2. Starting with the one of the most famous funds in the world, Berkshire Hathaway, We made 3 different scenarios in implementing portfolios with sector weightings- two theoretical and one real world:  ‘A portfolio without rebalancing’, ‘ with rebalancing quarterly’ and finally, ‘a real world portfolio with rebalancing quarterly, but lagging at least 45 days because of the time gap between the day 13F quarterly filings is published and the actual quarter end date. So we assumed we execute rebalancing upon we are able to have the most recent data. 

You can see the portfolio returns in comparison to the market return. We clearly found that portfolio with rebalancing performs best out of others. But Berkshire couldn’t beat the market return most of the times. So what we did from here was using the best rebalancing method, which is actively rebalancing with quarterly data, for all other candidate portfolio with optimized weightings. 

There were 6 different weightings; 4 from individual funds like Berkshire, GGHC, Soros and Cypress, while we created 2 optimized weightings by using methods like taking the average of 10 renowned funds quarterly weightings by sector, and pulling out the best quarterly weighting from a fund that performs well in the given quarter by using Whale score.

![overall performance](https://github.com/coolwonny/Chasing-the-Whale/blob/master/Images/Overall_performances.png)   

Here is the result, we plotted out each of portfolio return against market return. We didn’t find one portfolio that beat the market return all the time in the given period of time, but found out 4 out of 6 portfolios that performed well enough over the market in our observation.  The best one is Cypress followed by Best_quarterly, GGHC and Average. Cypress is heavily weighted at IT so we cannot just follow what they did in general, so we picked best_quarterly portfolio gives a certain intuition for getting the optimized sector weightings. 


Next part is using machine learning to forecast the performance after 30 days from now to make sure the optimized portfolio can have sustainable performance over market down the road. We used Linear Regression, ARIMA and SARIMAX and RNN LSTM model as our data is time-series. All the models predicted well enough with given test data. But the problem we had was how to forecast portfolio returns 30 days from now on.


The only model can give an answer to this was ARIMA and SARIMAX, cuz they do not need a certain test data to predict the future. So here’s the prediction from ARIMA. The implication is that we can expect our portfolio keep outperforming over market predicted return in the next 30 days, according to ARIMA model.

## Conclusion
Our research explains why we should not just follow the optimization method in the real world. 
Although we were able to demonstrate our portfolios outperforming (partially, not all the time) the market return, we may not be able to make it a real because of the time gap between the date of quarter ends and reporting due date which normally takes 45 days.     

However, we were successful in suggesting an idea that we could emulate the performance of hedge funds only holding 11 sector ETFs and its weightings to get the alpha over market returns. This idea will be able to evolve if we can add more theories and skills to generate a series of optimized sector weightings and its forecast by using machine learning methods. 



### Additional Resources
- [AQIS Fund II FAQ - March 2019](Additional%20Resources/AQIS%20Fund%20II%20FAQ%20-%20March%202019.pdf)
- [AQIS Strategy Review - 2019](Additional%20Resources/AQIS%20Strategy%20Review%20-%202019.pdf)
- [Python for Finance](Additional%20Resources/Python%20for%20Finance%20--%20Yves%20Hilpisch.pdf)


## Cross-Selling of Vehicle Insurance

#### What is Cross Selling means...?

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/1_.png?raw=true)


- Salespeople use cross-selling and upselling strategies to generate more business from an existing customer base.

- `Cross-selling` is the sales tactic whereby customers are enticed to buy items related or complementary to what they plan to purchase. 
   Cross-selling techniques include recommending, offering discounts on, and bundling related products. 
   Like upselling, the company seeks to earn more money per customer and increase perceived value by addressing and satisfying consumer needs. 

example: if a customer is about to buy a mobile phone, you could offer them a memory card, a phone case, or a protection plan.

- `Upselling`, also known as suggestive selling, is the practice of persuading customers to purchase an upgraded or more expensive version of a product or service. 
  The goal is to maximize profits and create a better experience for the customer. 
  That experience can translate into an increase in the customer's perceived value and an increased Customer Lifetime Value (CLV)—the total contribution a customer makes to a company.

example: customer looking for a mid-range smartwatch, but we are trying to sell premium watch like Apple Watch or Galaxy watch
- Cross-selling is perhaps one of the easiest ways to grow the business as they have already established a relationship with the client. Further, it is more profitable as the cost of acquiring a new customer is comparatively higher.

## Business Scenario

- Our client is an Insurance company that has provided Health Insurance to its customers now they need your help in building a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.

- Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue.

- Client is looking for an end result which a ML model, which can categorize customers into `Customer who are interested` & `Customer who are not interested`.

## Data understanding & EDA

- The data have  information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc.
- Lets understand each variables

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/2_.png?raw=true)


- Lets see sample data

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/3_.png?raw=true)

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/4_.png?raw=true)

- The data have 12 variables including target column.
- There is no null values and duplicates in the dataset
- There is 3.8 lakh records in the given dataset

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/5_.png?raw=true)

- There are high correlations in the dataset which we will see in detail later on.

Now lets see each column in detail

`Column - Gender`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/6_.png?raw=true)

- column have male and female category which shows almost same in frequency
- Male shows 54.1% and female is of 45.9% in the entire dataset


`Column - Age`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/7_.png?raw=true)

- Age column shows outliers towards the higher side of distribution

`Column - Driving license`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/8_.png?raw=true)

- 0 means customer doesn't have DL 1 means customer have DL
- The column can be dropped since 99.8% customer have DL.


`Column - Region_code`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/9_.png?raw=true)

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/10_.png?raw=true)

- shows 53 distinct regions code
- region code 28 shows most frequent

`Column - Previously_Insured`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/11_.png?raw=true)

- 1 means Customer already has Vehicle Insurance, 0 means Customer doesn't have Vehicle Insurance.
- Both the class have almost same frequency.

`Column - Vehicle_Age`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/12_.png?raw=true)

- Shows 3 classes, Vehicle age less than 1 year, 1-2 years, and more than 2 years.
- more than 2 years old vehicles have less frequency in dataset.

`Column - Vehicle_Damage`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/13_.png?raw=true)

- Shows 2 classes, vehicles with damage and without damage. Both of their frequencies are almost same.

`Column - Annual_Premium`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/14_.png?raw=true)

- showing 48838 distinct categories

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/15_.png?raw=true)

`Column - Policy_Sales_Channel`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/16_.png?raw=true)

- This is as Anonymised Code for the channel of outreaching to the customer
- Showing 155 distinct codes 

`Column - Vintage`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/17_.png?raw=true)

- Number of Days, Customer has been associated with the company
- The histogram shows as uniform distribution of dataset

`Column - Response`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/18_.png?raw=true)

- 1 means Customer is interested, 0 means Customer is not interested
- As per the above graph it is an imbalanced data

- Let's see the correlation plot

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/19_.png?raw=true)


#### Let's do Bi-Variate analysis between Independent columns with Target column

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/20_.jpg?raw=true)

#### Let's check outliers using boxplot

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/21_.png?raw=true)


### Observations from EDA:
- Age between `30 to 55 years` have higher chance of buying insurance
- `Male` having higher chance of buying the insurance
- `Region_code 0f 28` have high occurrence than other, and also have higher conversion rate than other Region code.
- Customers who don't have vehicle insurance have higher probability to buy insurance than who already have vehicle insurance. The correlation heat map also shows negative correlation with target variable.
- Customers who have vehicle age between `1-2 years` have more probability to buy insurance than customers who have vehicle age of less than a year and more than 2 years.
- Customers who had vehicle damages have more chance to buy insurance.
- Annul_premium amount is mostly with in `55k` and most popular premium amount is `2630`.
- Top 3 Popular sales channels are `152, 26, 124` . and top 10 popular channels covers `92%` of the data available.
- Sales channel `26 and 124` have higher customer conversion rate than any other sales channel.
- Even though `152` is the most frequent but conversion rates are very less comparing to `26 and 124`.
- `Policy sales channel` shows a weak negative correlation with `target` column.
- `Policy sales channel` and  `Previously_Insured column` shows correlation with Age column. which is a moderate multi-co-linearity in dataset.

## Feature Engineering

- Let's work on categorical features.
#### Trying to experiment with Binning on categorical columns using KBinsDiscretizer

column - Region_Code


![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/22_.jpg?raw=true)

column - Annual_Premium

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/23_.jpg?raw=true)

column - Vintage

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/24_.jpg?raw=true)

- From the above plots we are selecting `strategy='quantile'` for `Region_Code and Vintage` columns,
Selecting `strategy='kmeans'` for `Annual_Premium` column.


- On `Age` column I'm converting continuous value into 5 classes  
            
        1) Adolescence (15 to 20 years)
        2) Early_adulthood (21 to 30 years)
        3) Mid_life (31 to 39 years)
        4) Mature_adulthood (40 to 65 years)
        5) Late_adulthood (66 above)

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/25_.png?raw=true)

- On `Policy_Sales_Channel` column I'm taking only top 10 categories, Remaining categories are combining to a single class called `other`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/26_.png?raw=true)


## Model Building and Evaluation

- Used Logistic Regression, Random Forest, XGBoost and BalancedRandomForestClassifier initially
- Random Forest and Logistic Regression were getting biased towards majority class.
- SVC was taking more time for training, so due to lack of resource I've not used SVC.
- XGBoost and BalancedRandomForestClassifier gave better result compared to Random Forest and Logistic Regression classifiers.
- Results are below.

### 1) DummyClassifier

- Classification Report

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/27_.png?raw=true)

- Confusion Matrix

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/28_.png?raw=true)

- ROC Curve and precision-Recall Curve

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/29_.png?raw=true)


### 2) RandomForestClassifier

- Classification Report

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/30_.png?raw=true)

- Confusion Matrix

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/31_.png?raw=true)

- ROC Curve and precision-Recall Curve

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/32_.png?raw=true)


### 3) BalancedRandomForestClassifier

- Classification Report

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/33_.png?raw=true)

- Confusion Matrix

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/34_.png?raw=true)

- ROC Curve and precision-Recall Curve

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/35_.png?raw=true)

### 4) XGBoost

- Classification Report

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/36_.png?raw=true)

- Confusion Matrix

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/37_.png?raw=true)

- ROC Curve and precision-Recall Curve

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/38_.png?raw=true)


### 5) LogisticRegression

- Classification Report

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/39_.png?raw=true)

- Confusion Matrix

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/40_.png?raw=true)

- ROC Curve and precision-Recall Curve

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/41_.png?raw=true)
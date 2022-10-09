
## Cross Selling of Vehicle Insurance

#### What is Cross Selling means...?

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/1_.png?raw=true)


- Salespeople use cross-selling and upselling strategies to generate more business from an existing customer base.

- `Cross-selling` is the sales tactic whereby customers are enticed to buy items related or complementary to what they plan to purchase. 
   Cross-selling techniques include recommending, offering discounts on, and bundling related products. 
   Like upselling, the company seeks to earn more money per customer and increase perceived value by addressing and satisfying consumer needs. 

example: if a customer is about to buy a mobile phone, you could offer them a memory card, a phone case, or a protection plan.

- `Upselling`, also known as suggestive selling, is the practice of persuading customers to purchase an upgraded or more expensive version of a product or service. 
  The goal is to maximize profits and create a better experience for the customer. 
  That experience can translate into an increase in the customer's perceived value and an increased Customer Lifetime Value (CLV)—the total contribution a customer makes to a company.

example: customer looking for a mid-range smartwatch, but we are trying to sell premium watch like Apple watch or Galaxy watch
- Cross-selling is perhaps one of the easiest ways to grow the business as they have already established a relationship with the client. Further, it is more profitable as the cost of acquiring a new customer is comparatively higher.

## Business Scenario

- Our client is an Insurance company that has provided Health Insurance to its customers now they need your help in building a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.

- Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue.

- Client is looking for an end result which a ML model, which can categorize customers into `Customer who are interested` & `Customer who are not interested`.

## Data understanding & EDA

- The data have  information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc.
- Lets understand each variables

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/2_.png?raw=true)


- Lets see sample data

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/3_.png?raw=true)

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/4_.png?raw=true)

- The data have 12 variables including target column.
- There is no null values and duplicates in the dataset
- There is 3.8 lakh records in the given dataset

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/5_.png?raw=true)

- There are high correlations in the dataset which we will see in detail later on.

Now lets see each column in detail

`Column - Gender`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/6_.png?raw=true)

- column have male and female category which shows almost same in frequency
- Male shows 54.1% and female is of 45.9% in the entire dataset


`Column - Age`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/7_.png?raw=true)

- Age column shows outliers towards the higher side of distribution

`Column - Driving license`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/8_.png?raw=true)

- 0 means customer doesn't have DL 1 means customer have DL
- The column can be dropped since 99.8% customer have DL.


`Column - Region_code`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/9_.png?raw=true)

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/10_.png?raw=true)

- shows 53 distinct regions code
- region code 28 shows most frequent

`Column - Previously_Insured`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/11_.png?raw=true)

- 1 means Customer already has Vehicle Insurance, 0 means Customer doesn't have Vehicle Insurance.
- Both the class have almost same frequency.

`Column - Vehicle_Age`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/12_.png?raw=true)

- Shows 3 classes, Vehicle age less than 1 year, 1-2 years, and more than 2 years.
- more than 2 years old vehicles have less frequency in dataset.

`Column - Vehicle_Damage`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/13_.png?raw=true)

- Shows 2 classes, vehicles with damage and without damage. Both of their frequencies are almost same.

`Column - Annual_Premium`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/14_.png?raw=true)

- showing 48838 distinct categories

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/15_.png?raw=true)

`Column - Policy_Sales_Channel`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/16_.png?raw=true)

- This is as Anonymised Code for the channel of outreaching to the customer
- Showing 155 distinct codes 

`Column - Vintage`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/17_.png?raw=true)

- Number of Days, Customer has been associated with the company
- The histogram shows as uniform distribution of dataset

`Column - Response`

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/18_.png?raw=true)

- 1 means Customer is interested, 0 means Customer is not interested
- As per the above graph it is an imbalanced data

- Let's see the correlation plot

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/19_.png?raw=true)


#### Let's do Bi-Variate analysis between Independent columns with Target column

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/20_.jpg?raw=true)

#### Let's check outliers using boxplot

![alt text](https://github.com/sudheeshe/Cross_Selling/blob/main/Images_for_readme/21_.png?raw=true)


### Observations from EDA:
- Age between `30 to 55 years` have higher chance of buying insurance
- `Male` having higher chance of buying the insurance
- `Region_code 0f 28` have high occurrence than other, and also have higher conversion rate than other Region code.
- Cutomers who doesn't have vehicle insurance have higher proability to buy insurance than who already have vehicle insurance. The correlaton heat map also shows negative correlation with target variable.
- Customers who have vehicle age between `1-2 years` have more probability to buy insurance than customers who have vehicle age of less than a year and more than 2 years.
- Customers who had vehicle damages have more chance to buy insurance.
- Annul_premium amount is mostly with in `55k` and most popular premium amount is `2630`.
- Top 3 Popular sales channels are `152, 26, 124` . and top 10 popular channels covers `92%` of the data available.
- Sales channel `26 and 124` have higher customer conversion rate than any other sales channel.
- Even though `152` is the most frequent but conversion rates are very less comparing to `26 and 124`.
- `Policy sales channel` shows a weak negative correlation with `target` column.
- `Policy sales channel` and  `Previously_Insured column` shows correlation with Age column. which is a moderate multi-colinearity in dataset.

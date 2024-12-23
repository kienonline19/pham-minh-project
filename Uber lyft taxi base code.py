
'''
import pyarrow.parquet as pq

ub = pq.read_table('fhvhv_tripdata_2024-03.parquet')
ub = ub.to_pandas()

ub.to_csv("fhvhv_tripdata_2024-03.csv", index=False)

yc = pq.read_table('yellow_tripdata_2024-03.parquet')
yc = yc.to_pandas()

yc.to_csv("yellow_tripdata_2024-03.csv", index=False)
'''

import pandas as pd
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg', force=True)
from matplotlib import pyplot as plt
import seaborn as sns


ub = pd.read_csv("uber_24_03_10P.csv",low_memory=False)

lt = pd.read_csv("lyft_24_03_10P.csv",low_memory=False)

yc = pd.read_csv("yellow_cab_24_03_10P.csv",low_memory=False)

ub.shape
ub.describe()
ub.info()
ub.columns

lt.shape
lt.describe()
lt.info()

yc.shape
yc.columns
yc.describe()

ub.isnull()
ub.isnull().sum() ## how many null/missing values
ub.isnull().sum().sum()

ub["PU_Borough"].value_counts()
ub["DO_Borough"].value_counts()

## deal with datatime

# request_datetime
# on_scene_datetime
# pickup_datetime
# dropoff_datetime

########################################
########################################

ub["request_datetime"] = pd.to_datetime(ub["request_datetime"])
ub["request_day"] = ub["request_datetime"].dt.day_name()
ub["request_hour"] = ub["request_datetime"].dt.hour
ub["request_minute"] = ub["request_datetime"].dt.minute
## ub["request_second"] = ub["request_datetime"].dt.second
ub["request_date"] = ub["request_datetime"].dt.day

## ub["request_date"]

ub["on_scene_datetime"] = pd.to_datetime(ub["on_scene_datetime"])
ub["pickup_datetime"] = pd.to_datetime(ub["pickup_datetime"])
ub["dropoff_datetime"] = pd.to_datetime(ub["dropoff_datetime"])

ub['on_scene_sec'] = (ub['on_scene_datetime'] - ub['request_datetime']).dt.total_seconds()
ub['pickup_sec'] = (ub['pickup_datetime'] - ub['request_datetime']).dt.total_seconds()
ub['dropoff_sec'] = (ub['dropoff_datetime'] - ub['pickup_datetime']).dt.total_seconds()

ub['on_scene_sec'].describe()
ub['pickup_sec'].describe()
ub['dropoff_sec'].describe()

ub["request_date"].value_counts()
ub["request_day"].value_counts()
ub["PU_Borough"].value_counts()

'''
Manhattan        597408
Brooklyn         410986
Queens           313454
Bronx            205347
Staten Island     21789
'''

ub.shape

ub = ub[(ub['on_scene_sec']>=0) & (ub['pickup_sec']>=0) & (ub['dropoff_sec']>=0)]

ub.shape

## ub = ub.drop(ub[(ub['on_scene_sec']<0) | (ub['pickup_sec']<0) | (ub['dropoff_sec']<0) ].index)


ub["total_fare"] = ub["base_passenger_fare"] + ub["tolls"] + ub["congestion_surcharge"] + ub["airport_fee"] + ub["tips"]

ub['Hourly_segments'] = ub.request_hour.map({0:'H1',1:'H1',2:'H1',3:'H1',4:'H2',5:'H2',6:'H2',7:'H2',8:'H3',
                                     9:'H3',10:'H3',11:'H3',12:'H4',13:'H4',14:'H4',15:'H4',16:'H5',
                                     17:'H5',18:'H5',19:'H5',20:'H6',21:'H6',22:'H6',23:'H6'})

########################################
### figures
sns.countplot(x=ub['PU_Borough'], data=ub)
plt.title("uber rides by region")
plt.xlabel("region")
plt.ylabel("frequency")
plt.show()

sns.countplot(y= "PU_Borough" ,data=ub)

sns.barplot(x=ub["PU_Borough"], y=ub["base_passenger_fare"], data=ub)
plt.show()

sns.barplot(x=ub[ub['PU_Borough']=='Manhattan'].request_hour, y=ub["pickup_sec"], data=ub)
plt.show()

sns.barplot(x=ub[(ub['PU_Borough']=='Manhattan') & (ub['request_day']=="Monday")].request_hour, y=ub["pickup_sec"], data=ub)
plt.show()

sns.barplot(x=ub[ub['PU_Borough']=='Manhattan'].request_day, y=ub["pickup_sec"], data=ub)
plt.show()

sns.barplot(x=ub["request_day"], y=ub["base_passenger_fare"], data=ub)
plt.show()

sns.barplot(x=ub["request_day"], y=ub["congestion_surcharge"], data=ub)
plt.show()

sns.barplot(x=ub["request_hour"], y=ub["congestion_surcharge"], data=ub)
plt.show()

sns.barplot(x=ub["request_day"], y=ub["airport_fee"], data=ub)
plt.show()

sns.barplot(x=ub["request_day"], y=ub["on_scene_sec"], data=ub)
plt.show()

sns.barplot(x=ub["request_hour"], y=ub["on_scene_sec"], data=ub)
plt.show()

sns.barplot(x=ub["request_hour"], y=ub["trip_time"], data=ub)
plt.show()

sns.barplot(x=ub["request_hour"], y=ub["trip_miles"], data=ub)
plt.show()

fig, ax = plt.subplots(figsize=(15,6))
sns.barplot(x=ub["PU_Borough"], y=ub["base_passenger_fare"], hue=ub["request_day"], data=ub)
plt.legend(loc='upper right', mode = "expand", ncol = 4)
plt.show()

plt.scatter(ub["base_passenger_fare"], ub["congestion_surcharge"])
plt.title("base_passenger_fare vs. congestion_surcharge")
plt.xlabel("base_passenger_fare")
plt.ylabel("congestion_surcharge")
plt.show()

## sns.boxplot(x="base_passenger_fare",y="congestion_surcharge",hue='PU_Borough',data=ub)
## plt.show()

plt.hist(ub['base_passenger_fare'], bins = 25)
plt.show()

plt.subplot(1,3,1) ## the first plot
plt.hist(ub['on_scene_sec'], bins = 25)
plt.subplot(1,3,2)
plt.hist(ub['pickup_sec'], bins = 25)
plt.subplot(1,3,3)
plt.hist(ub['dropoff_sec'], bins = 25)
plt.show()

g = sns.FacetGrid(ub, col="request_day")
g.map(plt.scatter, "trip_miles", "trip_time", alpha=.4)

g = sns.FacetGrid(ub, col="PU_Borough")
g.map(plt.scatter, "trip_miles", "trip_time", alpha=.4)

## plt.switch_backend('agg')
## mpl.use('TkAgg', force=True)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.figure(figsize=(20,20))
## for i,day in enumerate(ub["request_day"].unique()):
for i, day in enumerate(days):
    plt.subplot(3,3,i+1)
    plt.title("{}".format(day))
    ub[ub["request_day"]==day]["request_hour"].hist(bins='auto')

## list(enumerate(days))

plt.figure(figsize=(20,20))
for i, day in enumerate(days):
    plt.subplot(3,3,i+1)
    plt.title("{}".format(day))
    ub[(ub['PU_Borough']=='Manhattan') & (ub["request_day"]==day)]["request_hour"].hist(bins='auto')

########################################
########################################
## Aggregate the data by days and hours
ub_hours = ub.groupby(["PU_Borough","request_date","request_day","request_hour"], as_index = False).agg(
    avg_on_scene_sec = ('on_scene_sec', "mean"),
    avg_pickup_sec = ("pickup_sec", "count"),
    avg_dropoff_sec = ("dropoff_sec", "mean"),
    trip_count=("request_datetime", "count"),
    avg_distance=("trip_miles", "mean"),
    avg_trip_time=("trip_time", "mean"),
    avg_total_fare=("total_fare", "mean"),
    sum_total_fare=("total_fare", "sum"),
    avg_base_fare=("base_passenger_fare", "mean"),
    avg_congestion_surcharge=("congestion_surcharge", "mean"),
    avg_airport_fee=("airport_fee", "mean"),
    avg_driver_pay=("driver_pay", "mean")
)

plt.scatter(x = ub_hours[ub_hours.PU_Borough=='Manhattan'].avg_total_fare, y = ub_hours[ub_hours.PU_Borough=='Manhattan'].trip_count)
plt.show()

plt.scatter(x = ub_hours[ub_hours.request_day=="Monday"].avg_total_fare, y = ub_hours[ub_hours.request_day=="Monday"].trip_count)
plt.show()

sns.regplot(x = ub_hours[ub_hours.request_day=="Monday"].avg_total_fare, y = ub_hours[ub_hours.request_day=="Monday"].trip_count)
plt.show()

sns.regplot(x = ub_hours[(ub_hours.request_day=="Monday") & (ub_hours.PU_Borough=='Manhattan')].avg_total_fare,
            y = ub_hours[(ub_hours.request_day=="Monday") & (ub_hours.PU_Borough=='Manhattan')].trip_count)
plt.show()

sns.regplot(x = ub_hours[ub_hours.request_hour==7].avg_trip_time, y = ub_hours[ub_hours.request_hour==7].avg_distance)
plt.show()

sns.regplot(x = ub_hours[ub_hours.request_hour==7].avg_on_scene_sec, y = ub_hours[ub_hours.request_hour==7].trip_count)
plt.show()

sns.regplot(x = ub_hours[ub_hours.request_hour==7].avg_on_scene_sec, y = ub_hours[ub_hours.request_hour==7].avg_total_fare)
plt.show()

sns.regplot(x = ub_hours[ub_hours.request_day=="Monday"].avg_distance, y = ub_hours[ub_hours.request_day=="Monday"].trip_count)
plt.show()

sns.regplot(x = ub_hours[ub_hours.request_day=="Monday"].avg_congestion_surcharge, y = ub_hours[ub_hours.request_day=="Monday"].trip_count)
plt.show()


days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.figure(figsize=(15,15))
## for i,day in enumerate(ub["request_day"].unique()):
for i, day in enumerate(days):
    plt.subplot(3,3,i+1)
    plt.title(day, loc='right')
    sns.regplot(x = ub_hours[ub_hours.request_day==day].avg_total_fare, y = ub_hours[ub_hours.request_day==day].trip_count)


plt.figure(figsize=(15,15))
## for i,day in enumerate(ub["request_day"].unique()):
for i, day in enumerate(days):
    plt.subplot(3,3,i+1)
    plt.title(day, loc='right')
    sns.regplot(x = ub_hours[(ub_hours.PU_Borough=='Manhattan') & (ub_hours.request_day==day)].avg_total_fare,
                y = ub_hours[(ub_hours.PU_Borough=='Manhattan') & (ub_hours.request_day==day)].trip_count)

h = list(range(0,24))
plt.figure(figsize=(15,15))
for i, hour in enumerate(h):
    plt.subplot(6,4,i+1)
    plt.title(hour, fontsize=10, loc='right')
    sns.regplot(x = ub_hours[ub_hours.request_hour==hour].avg_total_fare, y = ub_hours[ub_hours.request_hour==hour].trip_count)


########################################
########################################
## Regression analysis
import statsmodels.formula.api as smf

reg1=smf.ols(formula=" avg_on_scene_sec ~ trip_count + C(request_day)+C(PU_Borough)",data=ub_hours[ub_hours.request_hour==7]).fit()
print(reg1.summary())

## export result as a csv
out=reg1.summary()
with open('reg1.csv','w') as f:
f.write(out.as_csv())

reg2=smf.ols(formula="avg_pickup_sec ~ avg_total_fare + C(request_day)+C(PU_Borough)",data=ub_hours[ub_hours.request_hour==7]).fit()
print(reg2.summary())

reg3=smf.ols(formula="avg_pickup_sec ~ avg_base_fare + C(request_day)+C(PU_Borough)",data=ub_hours[ub_hours.request_hour==7]).fit()
print(reg3.summary())

reg4=smf.ols(formula="avg_pickup_sec ~ avg_congestion_surcharge + C(request_day)+C(PU_Borough)",data=ub_hours[ub_hours.request_hour==7]).fit()
print(reg4.summary())

## export results of multiple regressions
from statsmodels.iolib.summary2 import summary_col

reg123 = summary_col([reg1,reg2,reg3,reg4])
print(reg123)
reg123.tables[0].to_csv("reg123_result.csv")


reg5=smf.ols(formula="trip_count ~ avg_base_fare + C(request_day)+C(PU_Borough)",data=ub_hours[ub_hours.request_hour==7]).fit()
print(reg5.summary())

reg6=smf.ols(formula="trip_count ~ avg_total_fare + C(request_day)+C(PU_Borough)",data=ub_hours[ub_hours.request_hour==7]).fit()
print(reg6.summary())

reg7=smf.ols(formula="trip_count ~ avg_congestion_surcharge + C(request_day)+C(PU_Borough)",data=ub_hours[ub_hours.request_hour==7]).fit()
print(reg7.summary())


########################################
########################################
##fares vary in 24 hours for each day

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for day in days:
##    mask = ub["request_day"] == day
    ub_day = ub[(ub['PU_Borough']=='Manhattan') & (ub["request_day"] == day)].groupby("request_hour", as_index=False).agg(avg_on_scene_sec = ('on_scene_sec', "mean"),
                                                                        avg_total_fare=("total_fare", "mean"),
                                                                        avg_base_fare=("base_passenger_fare", "mean"),
                                                                        avg_congestion_surcharge=("congestion_surcharge", "mean"),
                                                                        avg_pickup_sec = ("pickup_sec", "count"),
                                                                        avg_dropoff_sec = ("dropoff_sec", "mean"),
                                                                        trip_count = ("request_datetime", "count"),
                                                                        avg_distance = ("trip_miles", "mean"),
                                                                        avg_trip_time = ("trip_time", "mean"))
##    exec(f"ub_{day} = ub_day")
    f"ub_{day} = ub_day"

ub_weekday = ub[(ub["request_day"] != "Saturday") & (ub["request_day"] != "Sunday")].groupby("request_hour",as_index=False).agg(avg_on_scene_sec = ('on_scene_sec', "mean"),
                                                                        avg_total_fare=( "total_fare","mean"),
                                                                        avg_base_fare=("base_passenger_fare", "mean"),
                                                                        avg_congestion_surcharge=("congestion_surcharge", "mean"),
                                                                        avg_pickup_sec = ("pickup_sec", "count"),
                                                                        avg_dropoff_sec = ("dropoff_sec", "mean"),
                                                                        trip_count = ("request_datetime", "count"),
                                                                        avg_distance = ("trip_miles", "mean"),
                                                                        avg_trip_time = ("trip_time", "mean"))
ub_weekday["counted"] = ub_weekday["counted"] / 5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(ub_Monday.request_hour, ub_Monday.avg_trip_time, label='Monday')
ax1.plot(ub_Monday.request_hour, ub_Tuesday.avg_trip_time, label='Tuesday')
ax1.plot(ub_Monday.request_hour, ub_Wednesday.avg_trip_time, label='Wednesday')
ax1.plot(ub_Monday.request_hour, ub_Thursday.avg_trip_time, label='Thursday')
ax1.plot(ub_Monday.request_hour, ub_Friday.avg_trip_time, label='Friday')
ax1.set_title('Average waiting time for each hour of each days in a week')
ax1.set_xlabel('hours')
ax1.set_ylabel('trip time')
## ax1.set_xticks(np.arange(0, 24, 2))
## ax1.set_xticklabels(np.arange(0, 24, 2))
ax1.legend()

ax2.plot(ub_Monday.request_hour, ub_Monday.avg_distance, label='Monday')
ax2.plot(ub_Monday.request_hour, ub_Tuesday.avg_distance, label='Tuesday')
ax2.plot(ub_Monday.request_hour, ub_Wednesday.avg_distance, label='Wednesday')
ax2.plot(ub_Monday.request_hour, ub_Thursday.avg_distance, label='Thursday')
ax2.plot(ub_Monday.request_hour, ub_Friday.avg_distance, label='Friday')
ax2.set_title('Average trip miles for each hour of each days in a week')
ax2.set_xlabel('hours')
ax2.set_ylabel('trip miles')
ax2.legend()

fig, ax = plt.subplots()
ax.plot(ub_Monday.request_hour, ub_Monday.avg_on_scene_sec, label='Monday')
ax.plot(ub_Monday.request_hour, ub_Tuesday.avg_on_scene_sec, label='Tuesday')
ax.plot(ub_Monday.request_hour, ub_Wednesday.avg_on_scene_sec, label='Wednesday')
ax.plot(ub_Monday.request_hour, ub_Thursday.avg_on_scene_sec, label='Thursday')
ax.plot(ub_Monday.request_hour, ub_Friday.avg_on_scene_sec, label='Friday')
ax.set_title('Average trip miles for each hour of each days in a week')
ax.set_xlabel('hours')
ax.set_ylabel('waiting time')
ax.legend()

fig, ax = plt.subplots()
ax.plot(ub_Monday.request_hour, ub_Monday.counted, label='Monday')
ax.plot(ub_Monday.request_hour, ub_Tuesday.counted, label='Tuesday')
ax.plot(ub_Monday.request_hour, ub_Wednesday.counted, label='Wednesday')
ax.plot(ub_Monday.request_hour, ub_Thursday.counted, label='Thursday')
ax.plot(ub_Monday.request_hour, ub_Friday.counted, label='Friday')
ax.set_title('Average number of trips for each hour of each days in a week')
ax.set_xlabel('hours')
ax.set_ylabel('number of trips')
ax.legend()


##################################
##################################
## market competition

lt["request_datetime"] = pd.to_datetime(lt["request_datetime"])
lt["request_day"] = lt["request_datetime"].dt.day_name()
lt["request_hour"] = lt["request_datetime"].dt.hour
lt["request_minute"] = lt["request_datetime"].dt.minute
## lt["request_second"] = lt["request_datetime"].dt.second
lt["request_date"] = lt["request_datetime"].dt.day

## lt["request_date"]

lt["on_scene_datetime"] = pd.to_datetime(lt["on_scene_datetime"])
lt["pickup_datetime"] = pd.to_datetime(lt["pickup_datetime"])
lt["dropoff_datetime"] = pd.to_datetime(lt["dropoff_datetime"])

lt['on_scene_sec'] = (lt['on_scene_datetime'] - lt['request_datetime']).dt.total_seconds()
lt['pickup_sec'] = (lt['pickup_datetime'] - lt['request_datetime']).dt.total_seconds()
lt['dropoff_sec'] = (lt['dropoff_datetime'] - lt['pickup_datetime']).dt.total_seconds()

lt['on_scene_sec'].describe()
lt['pickup_sec'].describe()
lt['dropoff_sec'].describe()

lt["request_date"].value_counts()
lt["request_day"].value_counts()

lt.shape
lt = lt[(lt['pickup_sec']>=0) & (lt['dropoff_sec']>=0)]
lt.shape

lt["total_fare"] = lt["base_passenger_fare"] + lt["tolls"] + lt["congestion_surcharge"] + lt["airport_fee"] + lt["tips"]

## Aggregate the data by days and hours
lt_hours = lt.groupby(["PU_Borough","request_date","request_day","request_hour"], as_index = False).agg(
    avg_on_scene_sec = ('on_scene_sec', "mean"),
    avg_pickup_sec = ("pickup_sec", "count"),
    avg_dropoff_sec = ("dropoff_sec", "mean"),
    trip_count=("request_datetime", "count"),
    avg_distance=("trip_miles", "mean"),
    avg_trip_time=("trip_time", "mean"),
    avg_total_fare=("total_fare", "mean"),
    sum_total_fare=("total_fare", "sum"),
    avg_base_fare=("base_passenger_fare", "mean"),
    avg_congestion_surcharge=("congestion_surcharge", "mean"),
    avg_airport_fee=("airport_fee", "mean"),
    avg_driver_pay=("driver_pay", "mean")
)

##################################
yc['tpep_pickup_datetime'] = pd.to_datetime(yc['tpep_pickup_datetime'])
yc["request_day"] = yc['tpep_pickup_datetime'].dt.day_name()
yc["request_hour"] = yc['tpep_pickup_datetime'].dt.hour
yc["request_minute"] = yc['tpep_pickup_datetime'].dt.minute
yc["request_date"] = yc['tpep_pickup_datetime'].dt.day

yc['tpep_dropoff_datetime'] = pd.to_datetime(yc['tpep_dropoff_datetime'])

yc['dropoff_sec'] = (yc['tpep_dropoff_datetime'] - yc['tpep_pickup_datetime']).dt.total_seconds()

yc.shape
yc = yc[yc['dropoff_sec']>=0]
yc.shape

'''
yc.columns
Index(['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
       'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
       'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra',
       'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
       'total_amount', 'congestion_surcharge', 'Airport_fee', 'PU_Borough',
       'PU_Zone', 'PU_service_zone', 'DO_Borough', 'DO_Zone',
       'DO_service_zone'],
      dtype='object')
'''

yc["total_fare"] = yc["fare_amount"] + yc["tolls_amount"] + yc["congestion_surcharge"] + yc["improvement_surcharge"] + yc["Airport_fee"] + yc["tip_amount"]

yc_hours = yc[yc['PU_Borough']=='Manhattan'].groupby(["PU_Borough","request_date","request_day","request_hour"], as_index = False).agg(
    avg_dropoff_sec = ("dropoff_sec", "mean"),
    trip_count=("tpep_pickup_datetime", "count"),
    avg_distance=("trip_distance", "mean"),
    avg_trip_time=("dropoff_sec", "mean"),
    avg_total_fare=("total_fare", "mean"),
    sum_total_fare=("total_fare", "sum"),
    avg_base_fare=("fare_amount", "mean"),
    avg_congestion_surcharge=("congestion_surcharge", "mean"),
    avg_improvement_surcharge=("improvement_surcharge", "mean"),
    avg_airport_fee=("Airport_fee", "mean"),
)

yc_grouped = yc_hours["avg_total_fare"].groupby(yc_hours["request_hour"])

yc_grouped_count = yc_grouped.count().reset_index()
yc_grouped_sum = yc_grouped.sum().reset_index()
yc_grouped_mean = yc_grouped.mean().reset_index()

lt_grouped = lt_hours["avg_total_fare"].groupby(lt_hours["request_hour"])

lt_grouped_count = lt_grouped.count().reset_index()
lt_grouped_sum = lt_grouped.sum().reset_index()
lt_grouped_mean = lt_grouped.mean().reset_index()

ub_grouped = ub_hours["avg_total_fare"].groupby(ub_hours["request_hour"])

ub_grouped_count = ub_grouped.count().reset_index()
ub_grouped_sum = ub_grouped.sum().reset_index()
ub_grouped_mean = ub_grouped.mean().reset_index()

## competition plots
h = list(range(0,24))
fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.plot(h, ub_grouped.mean(), label='Uber')
ax1.plot(h, lt_grouped.mean(), label='Lyft')
ax1.plot(h, yc_grouped.mean(), label='Taxi')
ax1.set_title('Average fares for each hour of each days in a week')
ax1.set_xlabel('hours')
ax1.set_ylabel('fares')
## ax1.set_xticks(np.arange(0, 24, 2))
## ax1.set_xticklabels(np.arange(0, 24, 2))
ax1.legend()

ax2.plot(h, ub_grouped.count(), label='Uber')
ax2.plot(h, lt_grouped.count(), label='Lyft')
ax2.plot(h, yc_grouped.count(), label='Taxi')
ax2.set_title('Number of trips for each hour of each days in a week')
ax2.set_xlabel('hours')
ax2.set_ylabel('Number of trips')
## ax1.set_xticks(np.arange(0, 24, 2))
## ax1.set_xticklabels(np.arange(0, 24, 2))
ax2.legend()

ax3.plot(h, ub_grouped.sum(), label='Uber')
ax3.plot(h, lt_grouped.sum(), label='Lyft')
ax3.plot(h, yc_grouped.sum(), label='Taxi')
ax3.set_title('Sales each hour of each days in a week')
ax3.set_xlabel('hours')
ax3.set_ylabel('Sales')
## ax1.set_xticks(np.arange(0, 24, 2))
## ax1.set_xticklabels(np.arange(0, 24, 2))
ax3.legend()




















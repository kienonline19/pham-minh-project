import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


class RideSharingAnalysis:
    def __init__(self):
        # Load datasets
        self.uber_data = pd.read_csv("uber_24_03_10P.csv", low_memory=False)
        self.lyft_data = pd.read_csv("lyft_24_03_10P.csv", low_memory=False)
        self.taxi_data = pd.read_csv(
            "yellow_cab_24_03_10P.csv", low_memory=False)

    def preprocess_all_data(self):
        """Preprocess all datasets"""
        # Preprocess Uber data
        self.preprocess_uber_data()
        # Preprocess Lyft data
        self.preprocess_lyft_data()
        # Preprocess Taxi data
        self.preprocess_taxi_data()

    def preprocess_uber_data(self):
        """Preprocess Uber data with datetime conversions and calculations"""
        # Convert datetime columns
        datetime_columns = ['request_datetime', 'on_scene_datetime',
                            'pickup_datetime', 'dropoff_datetime']

        for col in datetime_columns:
            self.uber_data[col] = pd.to_datetime(self.uber_data[col])

        # Extract time components
        self.uber_data["request_day"] = self.uber_data["request_datetime"].dt.day_name()
        self.uber_data["request_hour"] = self.uber_data["request_datetime"].dt.hour
        self.uber_data["request_date"] = self.uber_data["request_datetime"].dt.day

        # Calculate time differences
        self.uber_data['on_scene_sec'] = (self.uber_data['on_scene_datetime'] -
                                          self.uber_data['request_datetime']).dt.total_seconds()
        self.uber_data['pickup_sec'] = (self.uber_data['pickup_datetime'] -
                                        self.uber_data['request_datetime']).dt.total_seconds()
        self.uber_data['dropoff_sec'] = (self.uber_data['dropoff_datetime'] -
                                         self.uber_data['pickup_datetime']).dt.total_seconds()

        # Filter invalid records
        self.uber_data = self.uber_data[
            (self.uber_data['on_scene_sec'] >= 0) &
            (self.uber_data['pickup_sec'] >= 0) &
            (self.uber_data['dropoff_sec'] >= 0)
        ]

        # Calculate total fare
        self.uber_data["total_fare"] = (self.uber_data["base_passenger_fare"] +
                                        self.uber_data["tolls"] +
                                        self.uber_data["congestion_surcharge"] +
                                        self.uber_data["airport_fee"] +
                                        self.uber_data["tips"])

    def preprocess_lyft_data(self):
        """Preprocess Lyft data with datetime conversions and calculations"""
        # Convert datetime columns
        datetime_columns = ['request_datetime', 'on_scene_datetime',
                            'pickup_datetime', 'dropoff_datetime']

        for col in datetime_columns:
            self.lyft_data[col] = pd.to_datetime(self.lyft_data[col])

        # Extract time components
        self.lyft_data["request_day"] = self.lyft_data["request_datetime"].dt.day_name()
        self.lyft_data["request_hour"] = self.lyft_data["request_datetime"].dt.hour
        self.lyft_data["request_date"] = self.lyft_data["request_datetime"].dt.day

        # Calculate time differences
        self.lyft_data['on_scene_sec'] = (self.lyft_data['on_scene_datetime'] -
                                          self.lyft_data['request_datetime']).dt.total_seconds()
        self.lyft_data['pickup_sec'] = (self.lyft_data['pickup_datetime'] -
                                        self.lyft_data['request_datetime']).dt.total_seconds()
        self.lyft_data['dropoff_sec'] = (self.lyft_data['dropoff_datetime'] -
                                         self.lyft_data['pickup_datetime']).dt.total_seconds()

        # Filter invalid records
        self.lyft_data = self.lyft_data[
            (self.lyft_data['pickup_sec'] >= 0) &
            (self.lyft_data['dropoff_sec'] >= 0)
        ]

        # Calculate total fare
        self.lyft_data["total_fare"] = (self.lyft_data["base_passenger_fare"] +
                                        self.lyft_data["tolls"] +
                                        self.lyft_data["congestion_surcharge"] +
                                        self.lyft_data["airport_fee"] +
                                        self.lyft_data["tips"])

    def preprocess_taxi_data(self):
        """Preprocess taxi data with datetime conversions and calculations"""
        # Convert datetime columns
        self.taxi_data['tpep_pickup_datetime'] = pd.to_datetime(
            self.taxi_data['tpep_pickup_datetime'])
        self.taxi_data['tpep_dropoff_datetime'] = pd.to_datetime(
            self.taxi_data['tpep_dropoff_datetime'])

        # Extract time components
        self.taxi_data["request_day"] = self.taxi_data['tpep_pickup_datetime'].dt.day_name()
        self.taxi_data["request_hour"] = self.taxi_data['tpep_pickup_datetime'].dt.hour
        self.taxi_data["request_date"] = self.taxi_data['tpep_pickup_datetime'].dt.day

        # Calculate time difference
        self.taxi_data['trip_duration'] = (self.taxi_data['tpep_dropoff_datetime'] -
                                           self.taxi_data['tpep_pickup_datetime']).dt.total_seconds()

        # Filter invalid records
        self.taxi_data = self.taxi_data[self.taxi_data['trip_duration'] >= 0]

        # Calculate total fare
        self.taxi_data["total_fare"] = (self.taxi_data["fare_amount"] +
                                        self.taxi_data["tolls_amount"] +
                                        self.taxi_data["congestion_surcharge"] +
                                        self.taxi_data["improvement_surcharge"] +
                                        self.taxi_data["Airport_fee"] +
                                        self.taxi_data["tip_amount"])

    def analyze_network_effects(self):
        """Analyze network effects through regional distribution and timing patterns"""
        # Regional distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.uber_data['PU_Borough'])
        plt.title("Uber Rides by Region")
        plt.xlabel("Region")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Time-based patterns
        plt.figure(figsize=(20, 20))
        days = ["Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"]
        for i, day in enumerate(days):
            plt.subplot(3, 3, i+1)
            plt.title(f"{day}")
            self.uber_data[self.uber_data["request_day"]
                           == day]["request_hour"].hist(bins='auto')
        plt.tight_layout()
        plt.show()

    def analyze_market_competition(self):
        """Analyze market competition between services"""
        # Aggregate hourly data for all services
        uber_hourly = self.uber_data.groupby("request_hour").agg({
            'total_fare': 'mean',
            'request_datetime': 'count'
        }).reset_index()

        lyft_hourly = self.lyft_data.groupby("request_hour").agg({
            'total_fare': 'mean',
            'request_datetime': 'count'
        }).reset_index()

        taxi_hourly = self.taxi_data.groupby("request_hour").agg({
            'total_fare': 'mean',
            'tpep_pickup_datetime': 'count'
        }).reset_index()

        # Plot comparisons
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Average fares
        ax1.plot(uber_hourly.request_hour,
                 uber_hourly.total_fare, label='Uber')
        ax1.plot(lyft_hourly.request_hour,
                 lyft_hourly.total_fare, label='Lyft')
        ax1.plot(taxi_hourly.request_hour,
                 taxi_hourly.total_fare, label='Taxi')
        ax1.set_title('Average Fares by Hour')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Average Fare')
        ax1.legend()

        # Trip volumes
        ax2.plot(uber_hourly.request_hour,
                 uber_hourly.request_datetime, label='Uber')
        ax2.plot(lyft_hourly.request_hour,
                 lyft_hourly.request_datetime, label='Lyft')
        ax2.plot(taxi_hourly.request_hour,
                 taxi_hourly.tpep_pickup_datetime, label='Taxi')
        ax2.set_title('Trip Volume by Hour')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Number of Trips')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def analyze_dynamic_pricing(self):
        """Analyze dynamic pricing patterns"""
        # Create hourly aggregations for all services
        uber_hours = self.uber_data.groupby(
            ["PU_Borough", "request_day", "request_hour"],
            as_index=False
        ).agg({
            'total_fare': ['mean', 'count'],
            'trip_miles': 'mean'
        })

        lyft_hours = self.lyft_data.groupby(
            ["PU_Borough", "request_day", "request_hour"],
            as_index=False
        ).agg({
            'total_fare': ['mean', 'count'],
            'trip_miles': 'mean'
        })

        # Plot pricing vs demand patterns
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=uber_hours, x=('total_fare', 'count'), y=('total_fare', 'mean'),
                        label='Uber', alpha=0.5)
        sns.scatterplot(data=lyft_hours, x=('total_fare', 'count'), y=('total_fare', 'mean'),
                        label='Lyft', alpha=0.5)
        plt.title('Relationship between Trip Volume and Pricing')
        plt.xlabel('Number of Trips')
        plt.ylabel('Average Fare')
        plt.legend()
        plt.show()

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        self.preprocess_all_data()
        self.analyze_network_effects()
        self.analyze_market_competition()
        self.analyze_dynamic_pricing()


if __name__ == "__main__":
    # Initialize and run analysis
    analysis = RideSharingAnalysis()
    analysis.run_full_analysis()

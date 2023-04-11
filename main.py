import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns 
import matplotlib.pylab as plt
import pickle
import joblib

st.title("Indian Flight Ticket Price Prediction")

flight_df = pd.read_csv("flight_data.csv")

st.subheader(f"Data Frame consists of {flight_df.shape[0]} rows and {flight_df.shape[1]} columns. ")

st.write("Here is the first 20 rows of the dataset:")

st.table(flight_df.head(20))

st.header("Let's do some data analysis")

st.write("Some of data analysis on source city and destination city")

grouped_by_cities = flight_df.groupby(["source_city","destination_city"]).price.mean()

fig, ax = plt.subplots()

ax.set_title("Average ticket price from one city to another.")

grouped_by_cities.plot(kind="barh", ax=ax)
ax.margins(0.3)

st.pyplot(fig)

fig2, ax2 = plt.subplots(2,1)

fig2.tight_layout(h_pad=4)

sns.countplot(data=flight_df, x="destination_city", ax=ax2[0])
sns.countplot(data=flight_df, x="source_city", ax=ax2[1])

ax2[0].set_title("Information about Destination city")
ax2[1].set_title("Information about Source city")

ax.margins(0.3)

st.pyplot(fig2)

st.write("Data analysis on other categorical columns")

fig3, ax3 = plt.subplots(4,1, figsize=(16,30))

sns.countplot(data=flight_df, x="class", ax=ax3[0])
sns.countplot(data=flight_df, x="departure_time", ax=ax3[1])
sns.countplot(data=flight_df, x="arrival_time", ax=ax3[2])
sns.countplot(data=flight_df, x="airline", ax=ax3[3])

ax3[0].set_title("Number of each type of class")
ax3[1].set_title("Number of each type of departure time")
ax3[2].set_title("Number of each type of arrival time")
ax3[3].set_title("Number of each type of airline")
 
st.pyplot(fig3)

st.write("Distribution Graphs")

fig4, ax4 = plt.subplots(3,1, figsize=(15,25))

sns.histplot(data=flight_df, x="price", hue="airline", ax=ax4[0]) 
sns.histplot(data=flight_df, x="price", hue="class", ax=ax4[1]) 
sns.histplot(data=flight_df, x="price", hue="stops", ax=ax4[2]) 

ax4[0].set_title("")
ax4[1].set_title("Number of each departure time")
ax4[2].set_title("Number of each arrival time")

st.pyplot(fig4)

st.subheader("Conclusion")

st.write("""As we saw in the above graphs it is more expensive to go from Bangalore to Kolkata than going from other cities to Bangalore.
    And it is cheaper to go from Hyderabad to Delhi than going from other cities to Delhi. And mostly people travel to Mumbay and
    Delhi from various cities. Therefore mostly people travel from those cities to other cities which makes a sense.
    Also mostly people fly in economy class rather than business class as it is cheaper to fly in economy than business class.
    Moreover people rarely fly in the late night. Moreover mostly passengers come to their destination city in the night, evening and morning.
    Also the most of people choose Vistara airline and Airindia airline even though they are more expensive than other airlines.  
""")

st.subheader("About prediction model:")
st.write("""I have used various regression algorithms such as Lasso, LinearRegression, DesicionTree and RandomForest. The best performed model was RandomForest.
            Accuracy was 97 percent on the test dataset. And mean squared error was almost 4000 (indian rupiah). 
        """)

st.subheader("Here is the 10 rows of test dataset. You can can check the price.")

st.table(flight_df.iloc[[14145,  444, 15816,  7941, 16491, 15330,  5608, 10494, 17596,
            17406]])

st.subheader("Below you can see approximate flight ticket price.")


airline = st.selectbox(
    'Choose the type of airline',
    tuple(flight_df["airline"].unique()))

source_city = st.selectbox(
    'Choose the source city',
    tuple(flight_df["source_city"].unique()))

dest_city = st.selectbox(
    'Choose the destination city',
    tuple(flight_df["destination_city"].unique()))

departure_time = st.selectbox(
    'Choose the departure time',
    tuple(flight_df["departure_time"].unique()))

class_type = st.selectbox(
    'Choose the class type',
    tuple(flight_df["class"].unique()))

days_left = st.number_input("In how many days do you want to fly?", min_value=1)

duration = flight_df.groupby(["source_city","destination_city"])["duration"].mean().loc[(source_city, dest_city)]

button = st.button("Calculate price") 

pipe = joblib.load("pipe_flight_price_predictor.jbl")


if button:
    new_df = pd.DataFrame([[airline, source_city, departure_time, dest_city, class_type, duration, days_left]], columns=["airline","source_city","departure_time","destination_city","class","duration", "days_left"])
    prediction = pipe.predict(new_df)
    st.write(f"Price of ticket is {round(prediction[0])}")

st.subheader("Contact me:")
st.write("github: https://github.com/Abduqayyum")
st.write("email: rabduqayum@mail.ru")
st.write("linkedin: linkedin.com/in/abduqayum-rasulmukhamedov-70844624a")

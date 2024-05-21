# AstroData - UNDER MAJOR CONSTRUCTION/DEVELOPMENT !!!
Date to Astronomy data for Time Series

The purpose of this code is to provide with extra columns you may need for Machine Learning tasks that involve time series with a 1-day precision.
It converts a date to Astronomical data, that show location of the planets (constellations they are in) for a date of choice.
It also gives a distance of a planet to Sun, includes Moon phase, Earth's and Moon's distance from Sun, and Earth's tilt.

The constellation names are using BERT embeddings, since it is possible to have the names of constellations that were not encountered during the training. The embedding is resampled by the bicubic resample so that vector does not overflow the data-sets.

I could not use the position of objects in the sky, because that would imply a location on Earth. Instead, it had to be done against the constellations. This means, we are getting a less precise dataset, which may be enough, if combined with other plantets' data.
I have not tested if this data is enough for Machine Learning to guess the date (backward transformation), and we might need to add more data. It would be wise not to remove the dates from the train-set.
This approach may be considered astrology as well as astronomy.

Since using BERT embeddings, neural nets would be a good choice of ML due to possible constellations that were not in the training set but that might appear elsewhere...

FourierForest has been added to this small project, since it can be used for predicting the data relating to time-series. I hope to evolve this project into something I always wanted to have, which is a mobile application for dowsing based on time series and geographical data... so I might be posting more files here.

I will have to split the files and data gathering into the categories such as astral, earth, weather, etc. For now, I am thinking about adding these:

Static Data

Static data is information that does not change over time or changes very infrequently.
1. Geographical Data

    Address and Location Data:
        Country, city, state, and formatted address.
        Timezone information.
        Administrative regions and their codes (e.g., ISO codes).

    Elevation Data:
        Elevation above sea level.

    Geocoding Information:
        Reverse geocoding results (converting coordinates to addresses).

Predictable Data

Predictable data can be forecasted or estimated for future dates, often using models and historical data.
2. Celestial Data

    Astronomical Data:
        Sun and moon rise/set times: Can be predicted accurately for the next 7 or 14 days.
        Position of celestial bodies (sun, moon, planets, stars): Can be predicted accurately for the next 7 or 14 days.

3. Weather Data

    Current Weather:
        Temperature, humidity, wind speed/direction, precipitation, etc.: Can be predicted for the next 7 or 14 days, but the accuracy decreases with time.

    Historical Weather:
        Weather data for a specific timestamp in the past: Static data (historical).

    Weather Forecast:
        Forecast for future dates and times: Can be predicted for the next 7 or 14 days, with decreasing accuracy over longer periods.

4. Environmental Data

    Air Quality Data:
        Pollution levels (PM2.5, PM10, AQI): Can be predicted for the next few days based on models and current data, but long-term predictions (7-14 days) are less reliable.

    UV Index:
        Ultraviolet radiation levels: Can be predicted for the next 7 or 14 days, with fairly good accuracy.

5. Time-Dependent Events

    Day/Night Information:
        Sunrise and sunset times: Can be predicted accurately for the next 7 or 14 days.
        Twilight times: Can be predicted accurately for the next 7 or 14 days.

    Seasonal Information:
        Seasonal variation (e.g., summer, winter): Static data in terms of the calendar but can be used to understand seasonal trends over future periods.

APIs and Services

To obtain the above data, you can use the following APIs and services:

    Geographical and Weather Data:
        OpenCage Geocoder API for geographical and timezone data (static).
        OpenWeatherMap API for current weather, weather forecast (predictable).
        Weatherstack API for weather data (predictable).
        Dark Sky API (now integrated into Apple) for weather data (predictable).
        Climacell API for hyper-local weather data (predictable).
        National Weather Service API (US-specific) for weather data (predictable).

    Celestial Data:
        NASA API for astronomical data (predictable).
        Sunrise-Sunset API for sunrise and sunset times (predictable).
        Time and Date API for celestial information (predictable).

    Environmental Data:
        OpenAQ API for air quality data (predictable to some extent).
        Breezometer API for air quality data (predictable to some extent).
        EPA AirNow API (US-specific) for air quality data (predictable to some extent).


- Still not sure whether the data processing and predicting will be done on mobile device or on a PC. Gathering and alerts will 100% be done on device. I could test it with the earthquake and tremmors data.



# Android Device Data Readings

This document outlines the types of dynamic data that can be retrieved from various sources on an Android device. Data should be dynamic and predictable to some extent (minimum 7 days) if the user cannot control it, or static if the user can control it (e.g., altitude by changing the location).

## Celestial Objects

### Time Dependent, Location Independent (Solar System)
- Distance from the Sun
- Tilt (if any)
- Percentage completed of a year or rotation period around the Sun
- Percentage completed of a day or rotation period around the axis (if any)
- The phase (such as the moon phase)
- Length of a day (if not dependent on tilt and if day lengths vary)
- Seasonal information

### Time Dependent, Location Dependent
- Beyond Solar System:
  - Location in the sky
- Solar System:
  - Location in the sky

## Earth Only

### Geospatial Data
- Timestamp
- Longitude, Latitude
- Altitude
- Length of a day
- Sunset, Sunrise, Twilight
- Any timestamp broken into year, month, day,... or percentage assuming the maximum timestamp (e.g., 20 years into the future)

### Meteorological Data
- Weather, Humidity, Air Pressure
- Air quality
- UV Index

## Device Readings

- Speed and direction of travel
- Temperature
- Humidity
- Atmospheric pressure
- Accelerometer readings (movement, orientation)
- Gyroscope readings (rotation and twist)
- Magnetometer readings (magnetic field strength)
- Number of nearby networks (WiFi, Bluetooth)
- Battery level

## Notes

- Exclude data that is static and uncontrollable.
- Ensure necessary permissions are granted by the user to access certain data types.

## Usage

This information is intended to assist developers in creating applications that utilize dynamic data from various sources on Android devices.

## License

This project is licensed under the terms of the [MIT license](https://opensource.org/licenses/MIT).

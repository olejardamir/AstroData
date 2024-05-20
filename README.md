# AstroData
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

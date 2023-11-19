# AstroData
Date to Astronomy data for Time Series

The purpose of this code is to provide with extra columns you may need for Machine Learning tasks.
It converts a date to Astronomical data, that show location of the planets (constellations they are in) for a date of choice.
It also gives a distance of a planet to Sun, includes Moon phase, Earth's and Moon's distance from Sun, and Earth's tilt.

The constellation names are using BERT embeddings, since it is possible to have the names of constellations that were not encountered during the training. The embedding is resampled by the bicubic resample so that vector does not overflow the data-sets.

I could not use the position of objects in the sky, because that would imply a location on Earth. Instead, it had to be done against the constellations.
I have not tested if this data is enough for Machine Learning to guess the date (backward tranformation), and we might need to add more data to be able to accomplish that without a difficulty.

Since using BERT embeddings, neural nets would be a good choice of ML due to constellations that were not in the training set...

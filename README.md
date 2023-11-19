# AstroData
Date to Astronomy data for Time Series

The purpose of this code is to provide you with extra columns you may need for Machine Learning tasks.
It converts a date to Astronomical data, that show location of the planets (constellations they are in) for a date of choice.
It also gives the distance of a planet to Sun, and includes Moon phase, Earth's and Moon's distance from Sun, and Earth's tilt.

The constellation names are using BERT embeddings, since it is possible to have the names of constellations that were not encountered during the training. The embedding is resampled by the bicubic resample so that vector does not overflow the data-sets.



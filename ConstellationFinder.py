import datetime
import time
import warnings

import astropy.units as u
import ephem
import numpy as np
import torch
from astroplan import Observer
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_constellation, EarthLocation, get_body
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from erfa import ErfaWarning
from pytz import timezone
from scipy.ndimage import zoom
from skyfield.api import load, Topos
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore", category=ErfaWarning)

solar_system_ephemeris.set('de432s')

class Constellation_Finder:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def get_constellation(self, ra, dec):
        sky_location = SkyCoord(ra, dec, unit=u.deg, frame='icrs')
        constellation = get_constellation(sky_location)
        return constellation

    def get_planet_constellations(self, unix_timestamp):
        time_date = Time(unix_timestamp, format='unix')

        celestial_bodies = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune',
                            'pluto', 'sun', 'moon', 'earth']

        celestial_body_constellations = {}

        for body in celestial_bodies:
            try:
                celestial_body = get_body(body, time_date)
                if body == 'moon':
                    moon_phase = self.get_moon_phase(unix_timestamp)
                    celestial_body_constellations[body.capitalize()] = {
                        'constellation': self.get_constellation(celestial_body.ra.deg, celestial_body.dec.deg),
                        'distance_to_sun': self.get_distance_to_sun(body, time_date),
                        'moon_phase': moon_phase
                    }
                else:
                    celestial_body_constellations[body.capitalize()] = {
                        'constellation': self.get_constellation(celestial_body.ra.deg, celestial_body.dec.deg),
                        'distance_to_sun': self.get_distance_to_sun(body, time_date)
                    }
            except Exception as e:
                print(f"The coordinates of {body} are not available: {str(e)}")

        return celestial_body_constellations

    def get_distance_to_sun(self, body, time):
        """Get the distance to the Sun from the specified body at the specified time."""
        celestial_body = get_body(body, time)
        distance = celestial_body.distance.to(u.au)
        return distance.value

    def get_distance_to_earth(self, time):
        """Get the distance to the Earth from the specified body at the specified time."""
        sun = get_body('sun', time)
        distance = sun.distance.to(u.au)
        return distance.value

    def get_moon_phase(self, unix_timestamp):
        time_date = Time(unix_timestamp, format='unix')
        observer = Observer(location=EarthLocation.of_site('greenwich'))
        moonphase = moon_illumination(time_date)
        return moonphase

    def get_tilt(self, timestamp):
        date = datetime.datetime.fromtimestamp(timestamp)
        next_equinox = ephem.next_equinox(date)
        prev_equinox = ephem.previous_equinox(date)
        days = (next_equinox.datetime() - prev_equinox.datetime()).days
        tilt = 23.44 * (1 - 2 * ((date - prev_equinox.datetime()).days / days))
        return tilt

    def bicubic_resample(self, original_array, target_elements):
        original_length = len(original_array)
        zoom_factor = target_elements / original_length
        # Perform bicubic resampling
        resampled_array = zoom(original_array, zoom_factor, order=3)
        return resampled_array

    def get_bert_embedding(self, input_text, model_name="bert-base-uncased"):
        # Tokenize the input text and get the BERT embeddings
        input_ids = self.tokenizer(input_text, return_tensors="pt")[
            "input_ids"]
        with torch.no_grad():
            output = self.model(input_ids)
            embeddings = output.last_hidden_state.mean(dim=1)  # Average over all tokens
        # Keep only the first 16 dimensions of the embeddings
        embeddings = self.bicubic_resample(embeddings[0].tolist(), 16)
        # Convert the embeddings to a comma-separated string
        embedding_str = ",".join(map(str, embeddings))
        return embedding_str

    def get_planet_phase(self, unix_timestamp, planet_name, longitude, latitude):
        planets = load('de421.bsp')
        ts = load.timescale()

        # Adding timezone information to the datetime object
        utc_datetime = datetime.datetime.utcfromtimestamp(unix_timestamp).replace(tzinfo=timezone('UTC'))
        time = ts.utc(utc_datetime)

        observer = planets['earth'] + Topos(longitude, latitude)  # Replace with your observer's location

        planet = planets[planet_name.upper() + ' BARYCENTER']  # Use the barycenter name
        astrometric = observer.at(time).observe(planet)
        apparent = astrometric.apparent()

        elongation = apparent.separation_from(astrometric)
        phase_angle = 180 - elongation.degrees

        return str(phase_angle)

    def getLine(self, timestamp, longitude, latitude):

        # timestamp = datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp()
        consts = self.get_planet_constellations(timestamp)
        tilt = self.get_tilt(timestamp)

        # Get day/night data
        day_night_data = self.get_day_night_data(timestamp)

        # Create the output line
        lin = (
                (
                        self.get_bert_embedding(consts["Mercury"]["constellation"])
                        + "," + str(consts["Mercury"]["distance_to_sun"])
                        + "," + self.get_planet_phase(timestamp, 'mercury', longitude, latitude)
                ) + (
                        "," + self.get_bert_embedding(consts["Venus"]["constellation"])
                        + "," + str(consts["Venus"]["distance_to_sun"])
                        + "," + self.get_planet_phase(timestamp, 'venus', longitude, latitude)
                ) + (
                        "," + self.get_bert_embedding(consts["Mars"]["constellation"])
                        + "," + str(consts["Mars"]["distance_to_sun"])
                        + "," + self.get_planet_phase(timestamp, 'mars', longitude, latitude)
                ) + (
                        "," + self.get_bert_embedding(consts["Jupiter"]["constellation"])
                        + "," + str(consts["Jupiter"]["distance_to_sun"])
                        + "," + self.get_planet_phase(timestamp, 'jupiter', longitude, latitude)
                ) + (
                        "," + self.get_bert_embedding(consts["Saturn"]["constellation"])
                        + "," + str(consts["Saturn"]["distance_to_sun"])
                        + "," + self.get_planet_phase(timestamp, 'saturn', longitude, latitude)
                ) + (
                        "," + self.get_bert_embedding(consts["Uranus"]["constellation"])
                        + "," + str(consts["Uranus"]["distance_to_sun"])
                        + "," + self.get_planet_phase(timestamp, 'uranus', longitude, latitude)
                ) + (
                        "," + self.get_bert_embedding(consts["Neptune"]["constellation"])
                        + "," + str(consts["Neptune"]["distance_to_sun"])
                        + "," + self.get_planet_phase(timestamp, 'neptune', longitude, latitude)
                ) + (
                        "," + self.get_bert_embedding(consts["Pluto"]["constellation"])
                        + "," + str(consts["Pluto"]["distance_to_sun"])
                ) + (
                        "," + self.get_bert_embedding(consts["Sun"]["constellation"])
                        + "," + str(consts["Sun"]["distance_to_sun"])
                ) + (
                        "," + self.get_bert_embedding(consts["Moon"]["constellation"])
                        + "," + self.get_bert_embedding(consts["Earth"]["constellation"])
                        + "," + str(consts["Moon"]["distance_to_sun"])
                        + "," + str(consts["Moon"]["moon_phase"])
                ) + (
                        "," + str(tilt)

                ) + (
                        "," + ",".join(map(str, day_night_data))
                )
        )
        return lin

    def rearrange_string(self, input_string):
        elements = input_string.split(',')
        strings = []
        numbers = []

        for element in elements:
            element = element.strip()
            if element.replace(".", "", 1).isdigit():
                numbers.append(element)
            else:
                strings.append(element)

        result = ','.join(strings + numbers)
        return result

    def get_day_length(self, object_name):
        DAY_LENGTHS = {
            'Mercury': 4222.6,
            'Venus': 2802.0,
            'Earth': 24.0,
            'Mars': 24.6,
            'Jupiter': 9.9,
            'Saturn': 10.7,
            'Uranus': 17.2,
            'Neptune': 16.1,
            'Moon': 708.7,  # Earth's Moon
            'Pluto': 153.3,  # Dwarf planet
        }

        if object_name in DAY_LENGTHS:
            return DAY_LENGTHS[object_name]
        else:
            # Fetching data for celestial objects dynamically
            obj = Horizons(id=object_name, location='500@10', epochs={'start': '2024-01-01', 'stop': '2024-01-02', 'step': '1d'})
            eph = obj.ephemerides()
            rotation_period = eph['rot_per'][0]
            if np.isnan(rotation_period):
                raise ValueError(f"Day length for {object_name} is not available.")
            return rotation_period

    def calculate_day_night_info(self, object_name, unix_timestamp):
        day_length_hours = self.get_day_length(object_name)
        day_length_seconds = day_length_hours * 3600

        # Convert Unix timestamp to Julian Date
        time = Time(datetime.datetime.utcfromtimestamp(unix_timestamp), scale='utc')
        julian_date = time.jd

        # Calculate the position in the day/night cycle
        rotation_phase = (julian_date % (day_length_seconds / 86400)) * 86400 / day_length_seconds
        percentage_completed = (rotation_phase * 86400) / day_length_seconds

        # Determine if it's day or night (assuming a simplistic half-day, half-night model)
        if percentage_completed <= 50:
            period = 'day'
            percentage = percentage_completed * 2
        else:
            period = 'night'
            percentage = (percentage_completed - 50) * 2

        return percentage

    def get_day_night_data(self, unix_timestamp):
        ret = []
        for celestial_object in ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Moon', 'Pluto']:
            info = self.calculate_day_night_info(celestial_object, unix_timestamp)
            ret.append(info)
        return ret

    def timestamp_to_date(self, unix_timestamp):
        return datetime.datetime.utcfromtimestamp(unix_timestamp).strftime('%Y-%m-%d')


# Example Usage
usg = Constellation_Finder()
current_timestamp = int(time.time())
print(usg.getLine(current_timestamp, 0.0, 51.48))

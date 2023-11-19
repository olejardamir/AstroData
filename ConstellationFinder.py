import astropy.units as u
import ephem
from astroplan import Observer
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_constellation, EarthLocation, get_body
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time

solar_system_ephemeris.set('de432s')
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.ndimage import zoom
import datetime
import warnings
from erfa import ErfaWarning

warnings.filterwarnings("ignore", category=ErfaWarning)

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
                            'pluto', 'sun', 'moon']

        celestial_body_constellations = {}

        for body in celestial_bodies:
            try:
                celestial_body = get_body(body, time_date)
                celestial_body_constellations[body.capitalize()] = {
                    'constellation': self.get_constellation(celestial_body.ra.deg, celestial_body.dec.deg),
                    'distance_to_sun': self.get_distance_to_sun(body,
                                                                time_date) if body != 'sun' else self.get_distance_to_earth(
                        time_date)
                }
            except:
                print(f"The coordinates of {body} are not available")

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
        # Convert timestamp to datetime
        date = datetime.datetime.fromtimestamp(timestamp)

        # Get the 'next equinox' (either vernal or autumnal)
        next_equinox = ephem.next_equinox(date)

        # Get the 'previous equinox'
        prev_equinox = ephem.previous_equinox(date)

        # Calculate the days between the 'next' and 'previous' equinox
        days = (next_equinox.datetime() - prev_equinox.datetime()).days

        # Calculate the tilt
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

    def getLine(self, date_str):
        timestamp = datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp()
        consts = self.get_planet_constellations(timestamp)
        moon_phase = self.get_moon_phase(timestamp)
        tilt = self.get_tilt(timestamp)
        lin = (self.get_bert_embedding(consts["Mercury"]["constellation"]) + "," + str(
            consts["Mercury"]["distance_to_sun"]) + ","
               + self.get_bert_embedding(consts["Venus"]["constellation"]) + "," + str(
                    consts["Venus"]["distance_to_sun"]) + ","
               + self.get_bert_embedding(consts["Mars"]["constellation"]) + "," + str(
                    consts["Mars"]["distance_to_sun"]) + ","
               + self.get_bert_embedding(consts["Jupiter"]["constellation"]) + "," + str(
                    consts["Jupiter"]["distance_to_sun"]) + ","
               + self.get_bert_embedding(consts["Saturn"]["constellation"]) + "," + str(
                    consts["Saturn"]["distance_to_sun"]) + ","
               + self.get_bert_embedding(consts["Uranus"]["constellation"]) + "," + str(
                    consts["Uranus"]["distance_to_sun"]) + ","
               + self.get_bert_embedding(consts["Neptune"]["constellation"]) + "," + str(
                    consts["Neptune"]["distance_to_sun"]) + ","
               + self.get_bert_embedding(consts["Pluto"]["constellation"]) + "," + str(
                    consts["Pluto"]["distance_to_sun"]) + ","
               + self.get_bert_embedding(consts["Sun"]["constellation"]) + "," + str(
                    consts["Sun"]["distance_to_sun"]) + ","
               + self.get_bert_embedding(consts["Moon"]["constellation"]) + "," + str(
                    consts["Moon"]["distance_to_sun"]) + ","
               + str(moon_phase) + "," + str(tilt))
        return lin



# usg = Constellation_Finder()
# print(usg.getLine("2023-11-18"))

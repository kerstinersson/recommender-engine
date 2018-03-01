# evaluates the content based recommender engine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# import utitily functions
from utilities import *
from recommender_cat import *

# ten wines to test
test = ['7904', '7424',]
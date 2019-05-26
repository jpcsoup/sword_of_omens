# data exploration and testing

import numpy as np
import pandas as pd

gscores = pd.read_csv('ml-20m/genome-scores.csv')
gtags = pd.read_csv('ml-20m/genome-tags.csv')
links = pd.read_csv('ml-20m/links.csv')
movies = pd.read_csv('ml-20m/movies.csv')
ratings = pd.read_csv('ml-20m/ratings.csv')
tags = pd.read_csv('ml-20m/tags.csv')

gscores.shape #(11709768, 3)
gtags.shape #(1128, 2)
links.shape #(27278, 3)
movies.shape #(27278, 3)
ratings.shape #(20000263, 4)
tags.shape #(465564, 4)

tags.columns.tolist()
tags['tag'].nunique() #38643
tags['movieId'].nunique() #19545
# average of 1.97 movies per tag




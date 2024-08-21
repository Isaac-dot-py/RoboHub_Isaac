import os
from dotenv import load_dotenv

import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embeddings(text):
    if isinstance(text, str):
        return np.random.rand(512)

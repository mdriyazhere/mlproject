import sys
import os
import pandas as pd
from src.pipeline.exception import CustomException
from src.pipeline.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            # prefer the correctly-spelled preprocessor filename, but fall back to
            # the existing (misspelled) 'proprocessor.pkl' if present in artifacts
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            alt_preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)

            if os.path.exists(preprocessor_path):
                preprocessor = load_object(file_path=preprocessor_path)
            elif os.path.exists(alt_preprocessor_path):
                preprocessor = load_object(file_path=alt_preprocessor_path)
            else:
                # raise a clear error so CustomException will capture it
                raise FileNotFoundError(f"No preprocessor found. Checked: {preprocessor_path} and {alt_preprocessor_path}")
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

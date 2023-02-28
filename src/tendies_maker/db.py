import os

from sqlalchemy import create_engine


class DIO:
    def __init__(self):
        self.engine = create_engine(os.environ["SQLALCHEMY_DATABASE_URI"], echo=True)

    def write_data(self, data):

        return None

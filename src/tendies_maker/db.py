import os
from pathlib import Path

import dotenv
from sqlalchemy import create_engine


class DB:
    def __init__(self):
        try:
            basedir = Path(__file__).resolve().parents[2]
            dotenv_file = os.path.join(basedir, ".env")

            if os.path.isfile(dotenv_file):
                dotenv.load_dotenv(dotenv_file)
        except:
            pass

        self.engine = create_engine(os.environ["SQLALCHEMY_DATABASE_URI"], echo=True)

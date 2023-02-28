from loguru import logger

from ..datamodel import TrainingData

logger.info("Instantiate training data by scraping WSB, looking for 5% spike in last 30 days")
td = TrainingData(tgt_pct=0.05, tgt_days=30)
td.append_all()
logger.info("All data generated, writing to database")
td.write_data()
logger.info("Sucessfully written to db")
from src.tendies_maker.datamodel import TrainingData

td = TrainingData(tgt_pct=0.05, tgt_days=30)
td.append_all()
td.write_data()
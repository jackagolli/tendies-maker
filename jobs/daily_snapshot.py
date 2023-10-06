import datetime
from email.message import EmailMessage
import os
import smtplib

from pretty_html_table import build_table

from src.tendies_maker.gather import get_price_history, get_news, get_macro_econ_data, \
    get_options_snapshot
from src.tendies_maker.thinker import append_technical_indicators, news_sentiment_analysis, append_options_metrics, \
    append_fluctuations
from src.tendies_maker.datamodel import TrainingData as td

ticker = 'SPY'
price_history = get_price_history([ticker], 90)
price_history = price_history.sort_index(level=1)
price_history = append_technical_indicators(price_history)
price_history.dropna(inplace=True)
price_history, max_changes = append_fluctuations(price_history)
news = get_news(ticker, datetime.datetime.now().strftime("%Y-%m-%d"))
score, confidence = news_sentiment_analysis(news)
fomc_dates = td.scrape_fomc_calendar()
days_to_fomc = (fomc_dates[0] - datetime.datetime.today()).days
options_chain = get_options_snapshot(ticker)
options_chain, metrics = append_options_metrics(options_chain)

final_data = price_history.iloc[-1:, :].copy()
final_data.loc[:, 'news_score'] = score
final_data.loc[:, 'news_confidence'] = confidence
final_data.loc[:, 'days_to_fomc'] = days_to_fomc
final_data.loc[:, 'max_intraday_change'] = max_changes['value_of_max_intraday'][ticker]
final_data.loc[:, 'days_since_last_spike'] = max_changes['days_since_last_spike'][ticker]
final_data.loc[:, 'put_call_volume_ratio'] = metrics['put_call_ratio']
final_data.loc[:, 'weighted_avg_vwap'] = metrics['weighted_avg_vwap']

econ = get_macro_econ_data()

# Convert the DataFrame to HTML
transposed_data = final_data.transpose()
final_data_html = build_table(transposed_data, 'blue_light', index=True, float_format="{:0.3f}".format)
econ_data_html = build_table(econ, 'blue_light',  float_format="{:0.3f}".format)

sender_email = os.environ['FROM_EMAIL']
password = os.environ['EMAIL_SECRET']

msg = EmailMessage()
msg['From'] = "jagolli192@gmail.com"
# msg['To'] = ', '.join(["jagolli192@gmail.com", "rhehdgus10@gmail.com"])
msg['To'] = ', '.join(["jagolli192@gmail.com"])

msg['Subject'] = 'TendiesMaker Report'
html = f"""
<html>
  <head></head>
  <body>
    <h1>SPY Summary</h1>
    {final_data_html}

    <h1>Macroecon Data</h1>
    <h2>% change in each metric.</h2>
    {econ_data_html}
  </body>
</html>
"""

# csv_buffer = io.StringIO()
# msg.add_attachment(csv_buffer.getvalue(), filename=filename)
msg.set_content(html, subtype='html')
# Create secure connection with server and send email
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(sender_email, password)
    server.send_message(msg)

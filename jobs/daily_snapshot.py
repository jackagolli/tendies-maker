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
days = 90
news = get_news(ticker)
score = news_sentiment_analysis(news, datetime.datetime.now())

price_history = get_price_history([ticker], days)
price_history = price_history.sort_index(level=1)
price_history = append_technical_indicators(price_history)
price_history.dropna(inplace=True)
price_history, max_changes = append_fluctuations(price_history)

fomc_dates = td.scrape_fomc_calendar()
days_to_fomc = (fomc_dates[0] - datetime.datetime.today()).days
options_chain = get_options_snapshot(ticker)
options_chain, metrics = append_options_metrics(options_chain)

# pct_change_last_row = price_history.iloc[-2:, :].pct_change().iloc[-1:].copy()
# pct_change_last_row = pct_change_last_row.applymap(
#     lambda x: "{:.2f}%".format(x * 100) if pd.notna(x) else np.nan
# )

final_data = price_history.iloc[-1:, :].copy()

final_data.loc[:, 'news_score'] = score
final_data.loc[:, 'days_to_fomc'] = days_to_fomc
final_data.loc[:, 'max_intraday_change_90d'] = max_changes['value_of_max_intraday'][ticker]
final_data.loc[:, 'put_call_ratio'] = metrics['put_call_volume_ratio']
final_data.loc[:, 'options_avg_vwap'] = metrics['avg_vwap']

percent_cols = ['bbipband', 'intraday_change', 'news_score', 'news_confidence',
                'max_intraday_change_90d',
                'put_call_volume_ratio']

order_final = ['news_score', 'days_since_last_spike', 'days_to_fomc', 'put_call_ratio',
               'options_avg_vwap',
               'max_intraday_change_90d','tenkan_kijun_cross',
               'intraday_change', 'vwap', 'close',
               'high', 'open', 'volume',
               'trade_count',
               'ichi',  'price_vs_senkou_a', 'price_vs_senkou_b', 'rsi', 'bbipband', 'MACD_12_26']

econ = get_macro_econ_data()


def format_float(x):
    formatted = "{:.3f}".format(x)  # limit to 3 decimal places
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted


# Convert the decimals to percentages in the DataFrame
for col in final_data:
    if col in percent_cols:
        final_data[col] = (final_data[col] * 100).apply(
            lambda x: "{:.2f}%".format(x))  # converting to percentage and formatting as percentage string
    elif final_data[col].dtype == 'float64':
        final_data[col] = final_data[col].apply(format_float)  # apply custom formatting

# Do the same for the 'econ' DataFrame if necessary
econ['Latest % Change'] = econ['Latest % Change'] * 100
econ['Latest % Change'] = econ['Latest % Change'].apply(lambda x: "{:.2f}%".format(x))

# Convert the DataFrame to HTML
# final_data = pd.concat([final_data, pct_change_last_row], keys=['current_val', 'pct_change'])
final_data = final_data[order_final]
transposed_data = final_data.transpose()
final_data_html = build_table(transposed_data, 'blue_light', index=True)
econ_data_html = build_table(econ, 'blue_light', float_format="{:0.3f}".format, index=True)

sender_email = os.environ['FROM_EMAIL']
password = os.environ['EMAIL_SECRET']

msg = EmailMessage()
msg['From'] = "jagolli192@gmail.com"
msg['To'] = ', '.join(["jagolli192@gmail.com", "rhehdgus10@gmail.com"])
# msg['To'] = ', '.join(["jagolli192@gmail.com"])

msg['SubIts still showing coject'] = 'TendiesMaker Report'
html = f"""
<html>
  <head></head>
  <body>
    <h1>SPY Summary</h1>
    {final_data_html}

    <h1>Macroecon Data</h1>
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

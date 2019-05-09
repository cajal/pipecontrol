import schedule
import time
import requests


def get_url(url):
	requests.get(url)


schedule.every().day.at("09:00").do(get_url, 'http://shikigami.ad.bcm.edu/api/v1/surgery/spawn_missing_data')
schedule.every().day.at("10:00").do(get_url, 'http://shikigami.ad.bcm.edu/api/v1/surgery/notification')
schedule.every().day.at("16:00").do(get_url, 'http://shikigami.ad.bcm.edu/api/v1/surgery/notification')


while True:
	schedule.run_pending()
	time.sleep(60)

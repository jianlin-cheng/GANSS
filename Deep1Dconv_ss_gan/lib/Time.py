import time

def start_watch(message):
	print message
	return time.time()

def stop_watch(time1):
	time2 = time.time()
	diff = time2-time1
	print diff

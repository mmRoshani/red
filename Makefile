run:
	python ./main.py

start-pro:
	ray metrics launch-prometheus

stop-pro:
	ray metrics shutdown-prometheus

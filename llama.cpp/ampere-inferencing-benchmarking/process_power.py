import sys
import csv

with open(sys.argv[1], "r") as f:
	results = f.readlines()
with open(sys.argv[2], "r") as f:
	power_readings = f.readlines()
power_readings = [row.rstrip().split(",") for row in power_readings[1:]]
power_readings = [[float(row[0]), float(row[1])] for row in power_readings]

idx = 0
def calc_avg_power(start, finish):
	global idx
	power_over_period = []
	for i, reading in enumerate(power_readings[idx:]):
		if finish > reading[1] > start:
			power_over_period.append(reading[0])
		elif reading[1] > finish:
			idx = i
			return sum(power_over_period) / len(power_over_period)
	assert False 	

output = [results[0].rstrip().split(",")[:-2]+["avg_power_watts"]]
for row in results[1:]:
	row = row.rstrip().split(",")
	output.append(row[:-2]+[calc_avg_power(float(row[-2]), float(row[-1])-5)])

with open(sys.argv[1], "w") as f:
	writer = csv.writer(f)
	for row in output:
		writer.writerow(row)


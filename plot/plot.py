import sys
import matplotlib.pyplot as plt

#Assuming data are in a CSV file and the first row is column names
if len(sys.argv) < 3:
    print("Need at least one column to plot")
    exit()
file_name = sys.argv[1]
columns = [int (i) for i in sys.argv[2:]]
file_lines = open(file_name).readlines()
header = file_lines[0]
titles = [header.split(',')[i] for i in columns]
print(columns)
print(titles)
#By default we plot all the columns in the same graph
data = []
for i in columns:
    data.append([float(line.split(',')[i]) for line in file_lines[1:]])

for i in range(len(data)):
    plt.plot(data[i], label = titles[i])


legend = plt.legend()
# plt.ylim(27,34)
plt.show()

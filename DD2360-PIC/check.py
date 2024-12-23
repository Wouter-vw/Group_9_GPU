import matplotlib.pyplot as plt

# Read delta values from the text file
with open('delta_values.txt', 'r') as file:
    delta_values = [float(line.strip()) for line in file]
delta_values = [d for d in delta_values if d < 1]
# Plot the histogram
plt.hist(delta_values, bins=50, edgecolor='black')
plt.title('Histogram of Delta Values')
plt.xlabel('Delta Value')
plt.ylabel('Frequency')
plt.show()

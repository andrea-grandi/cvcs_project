import csv

# Define the file name
file_name = '../../infer_on_dataset/metadata.csv'

# Define the header and the data rows
header = ['file_name','text']

# Open the file in write mode
with open(file_name, mode='w', newline='') as file:
    #writer = csv.writer(file)
    
    # Write the header
    writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar=',')
    writer.writerow(header)

    # Write the data rows in a loop
    for image_number in range(265):
        row_1 = "output_detection_image_" + str(image_number)
        row_2 = "Player A is standing at and player B is"
        row = [row_1, row_2]
        writer.writerow(row)

print(f"Data has been written to {file_name}")

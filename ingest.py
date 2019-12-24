
"""
Ingests the data from several CSV files into
a sqlite database
"""

from csv import reader

import sqlite3 as sql

def insert_item(db, type_name, fields):
	"""
	Inserts the item into the database
	"""
	#lookup the table
	table = lookup_table(type_name)

	#make cursor
	cursor = db.cursor()

	sql = "insert into %s (item_code, name, year, value, unit) values (?, ?, ?, ?, ?);" % table

	#prepare the statement
	cursor.execute(sql, fields)


def lookup_table(element_type):
	"""
	Lookup the table associated with the element
	type
	"""
	results = None

	if element_type.startswith("Area harvested"):
		results = "harvest_area"
	
	elif element_type.startswith("Yield"):
		results = "yield"

	elif element_type.startswith("Production"):
		results = "production"

	elif element_type.startswith("Producer Price"):
		results = "producer_price"

	return results


def main():

	#constants
	#FILE_NAME = "NIG_Ag_Commodity_Price.csv"
	FILE_NAME = "NIG_Ag_Prod_Variables.csv"
	DB_NAME = "commodity.db"
	TYPE = 5
	ITEM_CODE = 6
	ITEM = 7
	YEAR = 8
	UNIT = 10
	VALUE = 11

	#connect to the database
	db = sql.connect(DB_NAME)

	#open the CSV file
	with open(FILE_NAME) as file_name:
		
		first = True

		#read the csv and insert each row
		for row in reader(file_name):

			if not first:
				element_type = row[TYPE]
				fields = [row[ITEM_CODE], row[ITEM], row[YEAR], row[VALUE], row[UNIT]]

				#insert the row
				insert_item(db, element_type, fields)

			first = False
		
		db.commit()

		db.close()



if __name__ == "__main__":
	main()

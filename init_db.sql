drop table producer_price;
drop table harvest_area;
drop table yield;
drop table production;

create table producer_price (
	item_code			integer,
	year					integer,
	name					text,
	value					real,
	unit					text,
	primary key (item_code, year, unit)
);

create table harvest_area (
	item_code			integer,
	year					integer,
	name					text,
	value					real,
	unit					text,
	primary key (item_code, year, unit)
);

create table yield (
	item_code			integer,
	year					integer,
	name					text,
	value					real,
	unit					text,
	primary key (item_code, year, unit)
);

create table production (
	item_code			integer,
	year					integer,
	name					text,
	value					real,
	unit					text,
	primary key (item_code, year, unit)
);
